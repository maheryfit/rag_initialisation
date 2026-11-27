"""Gradio web interface: scrape a page, index it in Chroma, and chat with it."""

import html
import sys
import logging
from functools import lru_cache
import gradio as gr

from modules.data_extraction import scrape_web_page, output_path
from modules.data_processing import load_documents
from modules.indexer import Indexer
from modules.llm_interface import change_llm_model, create_watsonx_embedding, create_watsonx_llm
from modules.query_engine import answer_user_query, source_urls, AUTO
from modules.retrievers import RETRIEVER_NAMES
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

embedding_model = lru_cache(maxsize=1)(create_watsonx_embedding)

# ponytail: one index shared by every browser tab (single local user, one app process).
# Use one root folder per user if several people use the app.
indexer = Indexer(config.CHROMA_DIR, load_documents, embedding_model, create_watsonx_llm)

# labels of modules.indexer.STEP_NAMES
STEPS = {
    "chunking": "Chargement des pages enregistrées (JSON)",
    "model": "Modèle d'embedding et découpage en blocs",
    "vector": "Embedding dans Chroma DB et index « vector »",
    "bm25": "Indexation « bm25 » (mots-clés, sans embedding)",
    "hybrid": "Retriever « hybrid » (fusion des scores vector et bm25)",
    "auto_merging": "Indexation « auto_merging » (Chroma DB)",
    "recursive": "Indexation « recursive » (Chroma DB)",
    "document_summary": "Indexation « document_summary » (résumés llama3.2 + Chroma DB)",
    "activate": "Activation de la nouvelle version",
}

# status -> (icon, French label, colour)
STATUSES = {
    "pending": ("○", "en attente", "#6b7280"),
    "running": ("⏳", "en cours", "#d97706"),
    "done": ("✅", "terminé", "#16a34a"),
    "failed": ("❌", "échec", "#dc2626"),
}


def render_status(status):
    """(HTML of the build steps, status message) for the dict given by ``Indexer.status``."""
    version = html.escape(status["version"] or "aucune")
    lines = [f"<p><strong>Version active :</strong> {version}</p>"]
    if status["queued"]:
        lines.append("<p>Une demande est en attente d'une autre construction.</p>")
    if not status["progress"]:
        lines.append("<p>Aucune construction depuis le démarrage.</p>")
        return _panel(lines), status["message"]

    items = []
    for key, label in STEPS.items():
        state, detail = status["progress"].get(key, ("pending", ""))
        icon, state_label, colour = STATUSES[state]
        # details can contain scraped URLs or error text, so they are escaped
        detail_html = f"<br><small>{html.escape(str(detail))}</small>" if detail else ""
        items.append(
            f'<li style="margin:6px 0;padding:8px 12px;border-left:4px solid {colour}">'
            f'<span aria-hidden="true">{icon}</span> <strong>{html.escape(label)}</strong> '
            f'<span style="color:{colour}">({state_label})</span>{detail_html}</li>'
        )
    lines.append(f'<ol style="list-style:none;padding:0">{"".join(items)}</ol>')
    return _panel(lines), status["message"]


def _panel(lines):
    return f'<div aria-live="polite"><h3>Construction de l’index</h3>{"".join(lines)}</div>'


def current_status():
    return render_status(indexer.status())


def load_index():
    """Load the saved index when the page opens. Never builds."""
    indexer.load()


def reembed(selected_model):
    """Ask for a new version of the index, built in the background."""
    change_llm_model(selected_model)
    indexer.request()
    return "⏳ Build requested. The current index stays available."


def process_url(url, selected_model):
    """Scrape a web page, save it as JSON, then ask for a new index version.

    Yields:
        Messages about the scraping. The build progress comes from ``current_status``.
    """
    url = (url or "").strip()
    if not url:
        yield "⚠️ Please enter a web page URL."
        return
    yield "⏳ Scraping..."
    data = scrape_web_page(url)
    if not data:
        yield (
            "❌ Nothing scraped. The URL may be invalid or not public, the site's robots.txt "
            "may disallow crawling (LinkedIn does), or the request failed. See the terminal log."
        )
        return
    reembed(selected_model)
    yield (
        f"✅ Saved to {output_path(url)}. Indexing runs in the background.\n\n"
        f"Title: {data['title']}\n\n{data['text'][:1000]}"
    )


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

    # read once: a build can switch to a new version while this question is answered
    knowledge_base = indexer.knowledge_base
    if knowledge_base is None:
        answer = "⚠️ No index ready. Click Re-embed, or process a URL."
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
                    process_btn = gr.Button("Process Page")
                    reembed_btn = gr.Button("Re-embed")
                    result_text = gr.Textbox(label="Result", lines=10)

                with gr.Column():
                    index_message = gr.Textbox(label="Index status", lines=3)
                    steps_html = gr.HTML(render_status(indexer.status())[0])

            process_btn.click(
                fn=process_url,
                inputs=[url_input, model_dropdown],
                outputs=[result_text],
            )
            reembed_btn.click(fn=reembed, inputs=[model_dropdown], outputs=[result_text])
            # runs when the page opens; does nothing if an index is already in memory
            demo.load(fn=load_index, show_progress="hidden")
            gr.Timer(1.0).tick(
                fn=current_status,
                outputs=[steps_html, index_message],
                show_progress="hidden",
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


def main():
    # Load the saved index before serving the page, so the chat is ready as soon as it opens.
    # This embeds no page and never builds. The wait is mostly the embedding model loading.
    logger.info("Loading the saved index...")
    indexer.load()
    demo = create_gradio_interface()
    # Local only: the scraper fetches any URL it is given, so do not expose the app
    # to the network without authentication.
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
