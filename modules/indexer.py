"""Build index versions in the background while the active one keeps answering."""

import logging
import threading
import time
from typing import Any, Callable, Optional

from modules import versions
from modules.retrievers import KnowledgeBase

logger = logging.getLogger(__name__)

STEP_NAMES = ["chunking", "model", "vector", "bm25", "hybrid", "auto_merging", "recursive",
              "document_summary", "activate"]
RETRIEVER_STEPS = STEP_NAMES[2:-1]


class Indexer:
    """Owns the knowledge base in use. One build runs at a time, in a thread."""

    def __init__(
        self,
        root: str,
        load_pages: Callable[[], list],
        embed_model: Callable[[], Any],
        llm: Callable[[], Any],
    ):
        self.root = root
        self._load_pages = load_pages
        self._embed_model = embed_model
        self._llm = llm
        self.knowledge_base: Optional[KnowledgeBase] = None
        self._state = {"running": False, "queued": False, "progress": {}, "message": "",
                       "version": None}
        self._lock = threading.Lock()  # protects _state
        self._disk_lock = threading.Lock()  # a load and a cleanup must not overlap
        self._idle = threading.Event()
        self._idle.set()

    def status(self) -> dict:
        with self._lock:
            return {**self._state, "progress": dict(self._state["progress"])}

    def wait(self, timeout: Optional[float] = 30) -> bool:
        return self._idle.wait(timeout)

    def _set(self, **values) -> None:
        with self._lock:
            self._state.update(values)

    def _step_status(self, step: str, status: str, detail: str = "") -> None:
        with self._lock:
            self._state["progress"][step] = (status, detail)

    def load(self) -> None:
        """Load the active version. Never builds and never raises."""
        with self._disk_lock:
            if self.knowledge_base is not None or self.status()["running"]:
                return
            try:
                versions.cleanup(self.root)
                message = self._load_active()
            except Exception as e:
                logger.exception("Loading the active version failed")
                message = f"❌ Cannot load the active version: {e}. Click Re-embed."
            self._set(message=message)

    def _load_active(self) -> str:
        active = versions.active_version(self.root)
        if active is None:
            return "No index yet. Click Re-embed, or process a URL."
        folder, data = active
        model = self._embed_model()
        if not versions.compatible(data, model.model_name):
            return ("⚠️ The saved index uses another embedding model or other chunk settings. "
                    "Click Re-embed.")
        self.knowledge_base = KnowledgeBase.load(str(folder), model, self._llm())
        self._set(version=folder.name)
        pending = versions.pending_pages(data, self._load_pages())
        if pending:
            return (f"⚠️ Index loaded, but {pending} page(s) changed since it was built. "
                    "Click Re-embed.")
        return "✅ Index loaded. Go to the Chat tab."

    def request(self) -> None:
        """Start a build, or ask for one more after the build under way."""
        with self._lock:
            if self._state["running"]:
                self._state["queued"] = True
                return
            self._state["running"] = True
            self._idle.clear()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            self._build()
            with self._lock:  # same lock as request(), so that no request is lost
                if not self._state["queued"]:
                    self._state["running"] = False
                    self._idle.set()
                    return
                self._state["queued"] = False

    def _build(self) -> None:
        """Build a new version folder and switch to it only if every step succeeds."""
        self._set(progress={}, message="⏳ Building a new version...")
        step, started = STEP_NAMES[0], time.perf_counter()

        def start(name):
            nonlocal step, started
            step, started = name, time.perf_counter()
            self._step_status(step, "running")

        def done(detail=""):
            elapsed = f"{time.perf_counter() - started:.1f} s"
            self._step_status(step, "done", f"{detail} ({elapsed})" if detail else elapsed)

        try:
            start("chunking")
            with self._disk_lock:
                versions.cleanup(self.root)
            documents = self._load_pages()
            if not documents:
                raise ValueError("No saved page. Process a URL first.")
            done(f"{len(documents)} page(s)")

            start("model")
            model = self._embed_model()
            previous = self._reusable_version(model.model_name)
            folder = versions.new_version_dir(self.root)
            knowledge_base = KnowledgeBase(
                documents, model, self._llm(), str(folder), previous_dir=previous
            )
            done(model.model_name)

            for name in RETRIEVER_STEPS:
                start(name)
                knowledge_base.retriever(name)
                done()

            start("activate")
            knowledge_base.save()
            versions.activate(self.root, folder, versions.manifest(documents, model.model_name))
            self.knowledge_base = knowledge_base
            done(folder.name)
            self._set(version=folder.name, message="✅ New version active. Go to the Chat tab.")
        except Exception as e:
            logger.exception("Build failed at step %s", step)
            self._step_status(step, "failed", str(e))
            kept = " The previous version stays active." if self.knowledge_base else ""
            self._set(message=f"❌ Build failed at step '{step}': {e}.{kept} Click Re-embed to retry.")

    def _reusable_version(self, model_name: str) -> Optional[str]:
        """Folder of the active version if its vectors can be reused."""
        try:
            active = versions.active_version(self.root)
        except ValueError as e:
            logger.warning("Nothing to reuse: %s", e)
            return None
        if active and versions.compatible(active[1], model_name):
            return str(active[0])
        return None
