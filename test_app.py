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


app.create_gradio_interface()
page = {"title": "T", "text": "hello"}

updates = run(page)
assert "en cours" in updates[0][0] and len(updates) > 5, "steps must be streamed"
final_html, message = updates[-1]
assert final_html.count("(terminé)") == 7 and final_html.count("(échec)") == 1, final_html
assert "&lt;boom&gt;" in final_html and "<boom>" not in final_html, "error text must be escaped"
assert "recursive" in message and app.state["knowledge_base"] is not None

assert run(page, build_all=False)[-1][0].count("(à la première utilisation)") == 4

failed_html = run({})[-1][0]
assert failed_html.count("(échec)") == 1 and failed_html.count("(en attente)") == 7, failed_html
print("app ok")
