"""Offline test of versioned indexing (no Ollama, no HF download). Run: python test_indexing.py"""
import re
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import chromadb
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import CompletionResponse, LLMMetadata, MockLLM
from llama_index.core.schema import Document

from modules import versions
from modules.indexer import Indexer, STEP_NAMES
from modules.retrievers import KnowledgeBase, RETRIEVER_NAMES


def page_document(topic, text=None):
    return Document(text=text or f"A page about {topic}.", doc_id=topic, metadata={"url": topic},
                    excluded_embed_metadata_keys=["url"], excluded_llm_metadata_keys=["url"])


def expect(error, function, *args):
    try:
        function(*args)
    except error:
        return
    raise AssertionError(f"{function.__name__} did not raise {error.__name__}")


def check_version_folders(root):
    assert versions.active_version(root) is None
    docs = [page_document("python")]
    first = versions.new_version_dir(root)
    versions.activate(root, first, versions.manifest(docs, "a"))
    folder, data = versions.active_version(root)
    assert folder == first and versions.compatible(data, "a") and not versions.compatible(data, "b")
    assert versions.pending_pages(data, docs) == 0
    assert versions.pending_pages(data, docs + [page_document("music")]) == 1
    assert versions.pending_pages(data, [page_document("music")]) == 2
    assert versions.pending_pages(data, [page_document("python", "changed")]) == 1

    # a failed pointer write leaves the old pointer intact
    second = versions.new_version_dir(root)
    with patch("os.replace", side_effect=OSError("disk full")):
        expect(OSError, versions.activate, root, second, versions.manifest(docs, "a"))
    assert versions.active_version(root)[0] == first

    # cleanup keeps active and previous, removes incomplete folders, leaves other content
    versions.activate(root, second, versions.manifest(docs, "a"))
    incomplete = versions.new_version_dir(root)
    Path(root, "chroma.sqlite3").write_text("legacy")
    Path(root, "vendor").mkdir()
    versions.cleanup(root)
    assert first.exists() and second.exists() and not incomplete.exists()
    assert Path(root, "chroma.sqlite3").exists() and Path(root, "vendor").exists()

    # a pointer to a missing folder, or a broken one, is reported, and cleanup deletes nothing
    for broken in ('{"active": "v1"}', "{broken", "[]", '{"active": "../x"}'):
        Path(root, "active.json").write_text(broken)
        expect(ValueError, versions.active_version, root)
        versions.cleanup(root)
        assert first.exists() and second.exists(), broken

    # activating over a broken pointer repairs it
    versions.activate(root, second, versions.manifest(docs, "a"))
    assert versions.active_version(root)[0] == second


class CountingEmbedding(MockEmbedding):
    """Records every page text it is asked to embed (questions are not recorded)."""

    embedded: list = []

    def _get_text_embeddings(self, texts):
        self.embedded.extend(texts)
        return super()._get_text_embeddings(texts)


class CountingLLM(MockLLM):
    """Counts calls and answers with the topics named in the prompt."""

    calls: int = 0

    def complete(self, prompt, formatted=False, **kwargs):
        self.calls += 1
        topics = sorted(set(re.findall(r"page about (\w+)", prompt)))
        return CompletionResponse(text=f"Summary: page about {' '.join(topics)}.")


def counting_models(model_name="a"):
    return CountingEmbedding(embed_dim=8, model_name=model_name, embedded=[]), CountingLLM()


def build(root, pages, previous=None, models=None):
    folder = versions.new_version_dir(root)
    documents = [page_document(*page) if isinstance(page, tuple) else page_document(page) for page in pages]
    KnowledgeBase(documents, *(models or counting_models()), str(folder),
                  previous_dir=str(previous) if previous else None).save()
    return folder


def snapshot(folder):
    client = chromadb.PersistentClient(path=str(folder))
    return {c.name: sorted(client.get_collection(c.name).get()["documents"])
            for c in client.list_collections()}


def all_texts(folder):
    return [text for documents in snapshot(folder).values() for text in documents]


def check_save_and_load(root):
    _, llm = models = counting_models()
    build(root, ["music"], models=models)
    calls_for_one_page = llm.calls
    assert calls_for_one_page > 0

    v1 = build(root, ["python", "cooking"])
    embed, llm = counting_models()
    kb = KnowledgeBase.load(str(v1), embed, llm)
    for name in RETRIEVER_NAMES:
        assert kb.retriever(name).retrieve("python"), name
    assert embed.embedded == [] and llm.calls == 0, (embed.embedded, llm.calls)  # restart costs nothing

    before = snapshot(v1)
    assert set(before) == {"chunks", "leaves", "children", "summaries"}, before
    embed, llm = models = counting_models()
    v2 = build(root, ["python", "cooking", "music"], previous=v1, models=models)
    assert embed.embedded and all("music" in text for text in embed.embedded), embed.embedded
    assert llm.calls == calls_for_one_page, llm.calls
    assert snapshot(v1) == before  # the previous version is untouched

    # removed pages leave nothing, neither in Chroma nor in the saved indexes
    v3 = build(root, ["python"], previous=v2)
    saved = [p.read_text(encoding="utf-8") for p in v3.rglob("*.json")]
    for text in all_texts(v3) + saved:
        assert "cooking" not in text and "music" not in text, text

    # a changed page is summarized and embedded again, and its old text is gone
    embed, llm = models = counting_models()
    v4 = build(root, [("python", "A page about rust.")], previous=v3, models=models)
    assert llm.calls == calls_for_one_page and embed.embedded, llm.calls
    assert all("python" not in text for text in all_texts(v4)), all_texts(v4)

    # a damaged version does not load
    chromadb.PersistentClient(path=str(v4)).delete_collection("leaves")
    expect(Exception, KnowledgeBase.load, str(v4), *counting_models())


def check_indexer(root):
    pages = [page_document("python"), page_document("cooking")]
    embed, llm = counting_models()

    def make_indexer(load_pages=lambda: list(pages), embed_model=lambda: embed, llm_factory=lambda: llm):
        return Indexer(root, load_pages, embed_model, llm_factory)

    def run(indexer):
        indexer.request()
        assert indexer.wait(), indexer.status()
        return indexer.status()

    def active():
        return versions.active_version(root)[0].name

    def failed(status):
        return [step for step, (state, _) in status["progress"].items() if state == "failed"]

    indexer = make_indexer()
    indexer.load()
    assert indexer.knowledge_base is None and "Re-embed" in indexer.status()["message"]
    status = run(indexer)
    old_kb, old_version = indexer.knowledge_base, status["version"]
    assert old_kb and old_version == active() and not failed(status), status
    assert list(status["progress"]) == STEP_NAMES, status["progress"]

    # any retriever failing keeps the old version
    for name in RETRIEVER_NAMES:
        with patch.object(KnowledgeBase, f"_build_{name}", side_effect=RuntimeError("boom")):
            status = run(indexer)
        assert indexer.knowledge_base is old_kb and active() == old_version, name
        assert failed(status) == [name] and status["version"] == old_version, (name, status)
        assert old_kb.retriever(name).retrieve("python"), name

    # so does a failed switch
    with patch.object(versions, "activate", side_effect=OSError("disk full")):
        status = run(indexer)
    assert indexer.knowledge_base is old_kb and active() == old_version
    assert failed(status) == ["activate"] and "disk full" in status["message"], status

    # restart: a new Indexer loads without building, and removes the incomplete folders
    embed.embedded.clear()
    llm.calls = 0
    incomplete = versions.new_version_dir(root)
    restarted = make_indexer()
    restarted.load()
    assert restarted.knowledge_base and not restarted.status()["running"]
    assert restarted.status()["version"] == old_version and not incomplete.exists()
    assert embed.embedded == [] and llm.calls == 0

    # pages saved since the active version are reported, not built
    pages.append(page_document("music"))
    restarted = make_indexer()
    restarted.load()
    assert "1 page" in restarted.status()["message"] and "Re-embed" in restarted.status()["message"]
    assert restarted.status()["version"] == old_version

    # no saved page: the active version is kept
    status = run(make_indexer(load_pages=lambda: []))
    assert "No saved page" in status["message"] and active() == old_version, status

    # another embedding model: the version is neither loaded nor deleted
    other = make_indexer(embed_model=lambda: counting_models("b")[0])
    other.load()
    assert other.knowledge_base is None and "Re-embed" in other.status()["message"]
    assert Path(root, old_version).is_dir()

    # the embedding model fails to load, then works
    attempts = []

    def flaky_model():
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("download failed")
        return embed

    flaky = make_indexer(embed_model=flaky_model)
    status = run(flaky)
    assert not status["running"] and failed(status) == ["model"] and active() == old_version, status
    assert not failed(run(flaky)) and active() != old_version

    # a broken pointer is reported, and a new build repairs it
    Path(root, "active.json").write_text("{broken")
    broken = make_indexer()
    broken.load()
    assert broken.knowledge_base is None and "Re-embed" in broken.status()["message"]
    assert not failed(run(broken)) and broken.knowledge_base


def check_queue(root):
    for first_fails in (False, True):
        release, runs = threading.Event(), []

        def load_pages():
            runs.append(1)
            if len(runs) == 1:
                assert release.wait(10)
                if first_fails:
                    raise RuntimeError("boom")
            return [page_document("python")]

        indexer = Indexer(root, load_pages, lambda: counting_models()[0], CountingLLM)
        for _ in range(3):
            indexer.request()
        assert indexer.status()["running"] and indexer.status()["queued"]
        release.set()
        assert indexer.wait() and len(runs) == 2, (first_fails, runs)
        assert indexer.knowledge_base and not indexer.status()["queued"]

    # a request made while a build is under way is not lost
    runs = []

    def llm_factory():
        runs.append(1)
        if len(runs) == 1:
            indexer.request()
        return CountingLLM()

    indexer = Indexer(root, lambda: [page_document("python")], lambda: counting_models()[0], llm_factory)
    indexer.request()
    assert indexer.wait() and len(runs) == 2, runs


with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    check_version_folders(tmp)

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    check_save_and_load(tmp)

def check_damaged_version_is_repaired(root):
    """Re-embed must work even if nothing can be reused from the active version."""
    indexer = Indexer(root, lambda: [page_document("python")], lambda: counting_models()[0], CountingLLM)

    def run():
        indexer.request()
        assert indexer.wait()
        return indexer.status()

    first = run()["version"]
    Path(root, first, "summaries.json").write_text('{"python": {"ha', encoding="utf-8")
    second = run()["version"]
    assert second != first, indexer.status()["message"]

    # a version whose Chroma database is not a database (never opened by this process)
    damaged = versions.new_version_dir(root)
    Path(damaged, "chroma.sqlite3").write_bytes(b"not a database" * 100)
    versions.activate(root, damaged, versions.manifest([page_document("python")], "a"))
    status = run()
    assert status["version"] not in (second, damaged.name), status["message"]


def check_old_versions_are_deleted(root):
    """Disk use must stay at two versions while the app runs, also on Windows."""
    indexer = Indexer(root, lambda: [page_document("python")], lambda: counting_models()[0], CountingLLM)
    for number in range(5):
        if number == 3:  # a failed build must not leave files locked either
            with patch.object(KnowledgeBase, "_build_document_summary", side_effect=RuntimeError("boom")):
                indexer.request()
                assert indexer.wait()
        indexer.request()
        assert indexer.wait()
    folders = sorted(p.name for p in Path(root).glob("v*"))
    assert len(folders) == 2 and indexer.status()["version"] in folders, folders
    assert indexer.knowledge_base.retriever("vector").retrieve("python")


def check_summaries_follow_the_llm(root):
    class OtherLLM(CountingLLM):
        @property
        def metadata(self):
            return LLMMetadata(model_name="other")

    v1 = build(root, ["python"])
    embed = counting_models()[0]
    llm = OtherLLM()
    build(root, ["python"], previous=v1, models=(embed, llm))
    assert llm.calls > 0, "a new LLM model must write new summaries"


for check in (check_indexer, check_queue, check_damaged_version_is_repaired,
              check_old_versions_are_deleted, check_summaries_follow_the_llm):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        check(tmp)

print("indexing ok")
