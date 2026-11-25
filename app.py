"""Gradio web interface: scrape a page, index it in Chroma, and chat with it."""

import html
import sys
import time
import logging
from functools import lru_cache
import gradio as gr

from modules.data_extraction import scrape_web_page, output_path
from modules.data_processing import load_documents
from modules.llm_interface import change_llm_model, create_watsonx_embedding, create_watsonx_llm
from modules.query_engine import answer_user_query, source_urls, AUTO
from modules.retrievers import KnowledgeBase, RETRIEVER_NAMES
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ponytail: one knowledge base shared by every browser tab (single local user).
# Key it by session, with separate Chroma collections, if several people use the app.
state = {"knowledge_base": None}

embedding_model = lru_cache(maxsize=1)(create_watsonx_embedding)


STEPS = {
    "scraping": "Scraping de la page et enregistrement en JSON",
    "chunking": "Chargement des JSON et découpage en blocs",
    "model": "Chargement du modèle d'embedding",
    "vector": "Embedding dans Chroma DB et index « vector »",
    "bm25": "Indexation « bm25 » (mots-clés, sans embedding)",
    "hybrid": "Retriever « hybrid » (fusion des scores vector et bm25)",
    "auto_merging": "Indexation « auto_merging » (Chroma DB)",
    "recursive": "Indexation « recursive » (Chroma DB)",
    "document_summary": "Indexation « document_summary » (résumés llama3.2 + Chroma DB)",
}

# status -> (icon, French label, colour)
STATUSES = {
    "pending": ("○", "en attente", "#6b7280"),
    "running": ("⏳", "en cours", "#d97706"),
    "done": ("✅", "terminé", "#16a34a"),
    "failed": ("❌", "échec", "#dc2626"),
    "skipped": ("⏭️", "à la première utilisation", "#6b7280"),
}


def render_steps(progress):
    """HTML list of the pipeline steps. ``progress`` maps a step to (status, detail)."""
    items = []
    for key, label in STEPS.items():
        status, detail = progress.get(key, ("pending", ""))
        icon, status_label, colour = STATUSES[status]
        # details can contain scraped URLs or error text, so they are escaped
        detail_html = f"<br><small>{html.escape(str(detail))}</small>" if detail else ""
        items.append(
            f'<li style="margin:6px 0;padding:8px 12px;border-left:4px solid {colour}">'
            f'<span aria-hidden="true">{icon}</span> <strong>{html.escape(label)}</strong> '
            f'<span style="color:{colour}">({status_label})</span>{detail_html}</li>'
        )
    return (
        '<div aria-live="polite"><h3>Étapes du pipeline</h3>'
        f'<ol style="list-style:none;padding:0">{"".join(items)}</ol></div>'
    )


def process_url(url, selected_model, build_all=True):
    """Scrape a web page, save it as JSON, and index all saved pages.

    Args:
        url: Web page URL to scrape.
        selected_model: LLM model to use.
        build_all: Build all retrievers now instead of on first use.

    Yields:
        (steps HTML, status message) each time a step starts or ends.
    """
    progress = {}
    url = (url or "").strip()
    if not url:
        yield render_steps(progress), "⚠️ Please enter a web page URL."
        return

    current, started = None, 0.0

    def start(key):
        nonlocal current, started
        current, started = key, time.perf_counter()
        progress[key] = ("running", "")
        return render_steps(progress), f"⏳ {STEPS[key]}..."

    def done(detail=""):
        elapsed = f"{time.perf_counter() - started:.1f} s"
        progress[current] = ("done", f"{detail} ({elapsed})" if detail else elapsed)

    try:
        yield start("scraping")
        data = scrape_web_page(url)
        if not data:
            progress[current] = ("failed", "Aucune donnée récupérée")
            yield render_steps(progress), (
                "❌ Nothing scraped. The URL may be invalid or not public, the site's robots.txt "
                "may disallow crawling (LinkedIn does), or the request failed. See the terminal log."
            )
            return
        done(f"{len(data['text'])} caractères dans {output_path(url)}")

        # the Chroma collections are about to be replaced, so the old retrievers are unusable
        state["knowledge_base"] = None
        change_llm_model(selected_model)

        yield start("chunking")
        documents = load_documents()
        done(f"{len(documents)} page(s)")

        yield start("model")
        model = embedding_model()
        done(config.LOCAL_EMBEDDING_MODEL_ID)

        yield start("vector")
        knowledge_base = KnowledgeBase(documents, model, create_watsonx_llm())
        knowledge_base.retriever("vector")
        done(f"{len(knowledge_base.chunks)} blocs dans {config.CHROMA_DIR}/")
    except Exception as e:
        logger.exception("Processing failed at step %s", current)
        progress[current] = ("failed", e)
        yield render_steps(progress), f"❌ Processing failed: {e}"
        return

    failed = []
    for name in [key for key in STEPS if key in RETRIEVER_NAMES and key != "vector"]:
        if not build_all:
            progress[name] = ("skipped", "")
            continue
        yield start(name)
        try:
            knowledge_base.retriever(name)
            done()
        except Exception as e:  # the other retrievers stay usable
            logger.exception("Building %s failed", name)
            progress[name] = ("failed", e)
            failed.append(name)

    state["knowledge_base"] = knowledge_base
    summary = f"⚠️ Failed retrievers: {', '.join(failed)}" if failed else "✅ Ready. Go to the Chat tab."
    yield render_steps(progress), f"{summary}\n\nTitle: {data['title']}\n\n{data['text'][:1000]}"


def plain_text(text):
    """Neutralise Markdown links, images and HTML in text influenced by scraped pages.

    Without this, a page could make the LLM write an image URL that leaks the chat
    to another server as soon as it is displayed.
    """
    return html.escape(str(text), quote=False).replace("[", "&#91;").replace("]", "&#93;")


def chat(user_query, retriever_name, chat_history):
    """Answer a question about the scraped pages.

    Args:
        user_query: User's question.
        retriever_name: A retriever name, or "auto" to use the router.
        chat_history: Chat history in Gradio's messages format.

    Returns:
        Updated chat history and an empty input box.
    """
    if not (user_query or "").strip():
        return chat_history, ""

    knowledge_base = state["knowledge_base"]
    if knowledge_base is None:
        answer = "⚠️ No page indexed. Please process a URL first."
    else:
        try:
            response, used = answer_user_query(knowledge_base, user_query, retriever_name)
            sources = "\n".join(f"- {u}" for u in source_urls(response))
            answer = f"{plain_text(response)}\n\n**Retriever:** `{used}`\n\n**Sources:**\n{sources}"
        except Exception as e:
            logger.exception("Query failed")
            answer = f"❌ Query failed: {plain_text(e)}"

    return chat_history + [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": answer},
    ], ""


def create_gradio_interface():
    """Create the Gradio interface."""
    # Define available LLM models
    available_models = [config.LLM_MODEL_ID]  # Ollama models; add more after `ollama pull`

    with gr.Blocks(title="Web Page RAG") as demo:
        gr.Markdown("# Web Page RAG")
        gr.Markdown("Scrape web pages, then ask questions in French or English")

        with gr.Tab("Process Web Page"):
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(
                        label="Web Page URL",
                        placeholder="https://quotes.toscrape.com/"
                    )
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        label="Select LLM Model",
                        value=config.LLM_MODEL_ID
                    )
                    build_all = gr.Checkbox(
                        label="Construire les 6 retrievers maintenant (plus long)",
                        value=True
                    )
                    process_btn = gr.Button("Process Page")

                with gr.Column():
                    steps_html = gr.HTML(render_steps({}))
                    result_text = gr.Textbox(label="Result", lines=10)

            process_btn.click(
                fn=process_url,
                inputs=[url_input, model_dropdown, build_all],
                outputs=[steps_html, result_text],
                show_progress="hidden"  # the steps panel shows the progress
            )

        with gr.Tab("Chat"):
            retriever_dropdown = gr.Dropdown(
                choices=[AUTO] + RETRIEVER_NAMES,
                label="Retriever (auto = query router)",
                value=AUTO
            )
            chatbot = gr.Chatbot(height=500)
            chat_input = gr.Textbox(
                label="Ask a question about the scraped pages",
                placeholder="De quoi parle cette page ?"
            )
            chat_btn = gr.Button("Send")

            for trigger in (chat_btn.click, chat_input.submit):
                trigger(
                    fn=chat,
                    inputs=[chat_input, retriever_dropdown, chatbot],
                    outputs=[chatbot, chat_input]
                )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    # Local only: the scraper fetches any URL it is given, so do not expose the app
    # to the network without authentication.
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
