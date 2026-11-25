"""Offline smoke test for modules/ (no Ollama, no HF download). Run: python smoke_test.py"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from scrapy.http import HtmlResponse

from modules import data_processing, query_engine, llm_interface, router
from modules.retrievers import KnowledgeBase, RETRIEVER_NAMES
from modules.web_spider import PageSpider, is_public_url

# the spider keeps visible text, drops scripts, and refuses local or non-http URLs
html = b"<title>T</title><body><h1>Hi <b>there</b></h1><script>bad()</script><p> some  text </p></body>"
page = next(PageSpider("https://example.com").parse(HtmlResponse("https://example.com", body=html)))
assert page["title"] == "T" and page["headings"] == ["Hi there"], page
assert "some text" in page["text"] and "bad()" not in page["text"], page
for blocked in ("file:///etc/passwd", "http://127.0.0.1:11434", "http://localhost", "http://192.168.1.1", "nonsense"):
    assert not is_public_url(blocked), blocked

# JSON -> documents -> Chroma -> all five retrievers -> answer
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    for i, topic in enumerate(["python", "cooking"]):
        text = " ".join(f"Sentence {n} talks about {topic}." for n in range(300))
        record = [{"url": f"https://example.com/{topic}", "title": topic, "description": "", "text": text}]
        Path(tmp, f"{i}.json").write_text(json.dumps(record), encoding="utf-8")
    documents = data_processing.load_documents(tmp)
    assert len(documents) == 2

    kb = KnowledgeBase(documents, MockEmbedding(embed_dim=8), MockLLM(max_tokens=20), str(Path(tmp, "chroma")))
    for name in RETRIEVER_NAMES:
        assert kb.retriever(name).retrieve("python"), name
        response, used = query_engine.answer_user_query(kb, "python", name)
        assert used == name and str(response).strip() and query_engine.source_urls(response), name
    assert {"chunks", "leaves", "children", "summaries"} <= {c.name for c in kb.client.list_collections()}
    # unparseable router output (MockLLM) must fall back instead of crashing
    assert query_engine.answer_user_query(kb, "python")[1] == router.DEFAULT_RETRIEVER

# the router maps the selector's choice to a retriever name, and rejects bad indexes
def selector_returning(i):
    result = SimpleNamespace(selections=[SimpleNamespace(index=i, reason="test")])
    return SimpleNamespace(select=lambda choices, query: result)

assert set(router.ROUTES) == set(RETRIEVER_NAMES)
for i, name in enumerate(router.ROUTES):
    assert router.route_query("q", None, selector_returning(i)) == name
assert router.route_query("q", None, selector_returning(99)) == router.DEFAULT_RETRIEVER

llm_interface.change_llm_model("llama3.2")
assert llm_interface.create_watsonx_llm().model == "llama3.2"
print("smoke ok")
