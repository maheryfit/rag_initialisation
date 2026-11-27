"""The six LlamaIndex retrievers, built over the scraped documents and Chroma DB."""

import json
import logging
import threading
from functools import cached_property
from pathlib import Path
from typing import List, Optional

import chromadb
from llama_index.core import (
    Document,
    DocumentSummaryIndex,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.base.response.schema import Response
from llama_index.core.indices.document_summary import DocumentSummaryIndexEmbeddingRetriever
from llama_index.core.indices.utils import embed_nodes
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever, RecursiveRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle, TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from modules.data_processing import chroma_storage, reuse_vectors
from modules.versions import text_hash
import config

logger = logging.getLogger(__name__)

RETRIEVER_NAMES = ["vector", "document_summary", "bm25", "auto_merging", "recursive", "hybrid"]


class HybridRetriever(BaseRetriever):
    """Weighted average of normalised vector and BM25 scores.

    Same method as "Exercise 1 - Build a Custom Hybrid Retriever" in the notebook.
    Both retrievers here share the same nodes, so results are matched by node ID.
    """

    def __init__(self, vector: BaseRetriever, bm25: BaseRetriever, vector_weight: float, top_k: int):
        self._weighted = [(vector, vector_weight), (bm25, 1 - vector_weight)]
        self._top_k = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes, scores = {}, {}
        for retriever, weight in self._weighted:
            results = retriever.retrieve(query_bundle)
            # BM25 scores are all 0 when no word matches, hence the "or 1"
            top = max((r.score or 0.0 for r in results), default=0.0) or 1.0
            for result in results:
                node_id = result.node.node_id
                nodes[node_id] = result.node
                scores[node_id] = scores.get(node_id, 0.0) + weight * (result.score or 0.0) / top
        best = sorted(scores, key=scores.get, reverse=True)[: self._top_k]
        return [NodeWithScore(node=nodes[i], score=scores[i]) for i in best]


class CachedSummarizer:
    """Response synthesizer that reuses the saved summary of an unchanged page."""

    def __init__(self, synthesizer, saved: dict, hashes: dict):
        self._synthesizer = synthesizer
        self._saved = saved  # {doc_id: {"hash": ..., "summary": ...}}
        self._hashes = hashes  # {doc_id: hash of the current text}

    def synthesize(self, query, nodes):
        doc_id = nodes[0].node.ref_doc_id
        saved = self._saved.get(doc_id)
        if saved and saved["hash"] == self._hashes[doc_id]:
            return Response(saved["summary"])
        return self._synthesizer.synthesize(query=query, nodes=nodes)


class KnowledgeBase:
    """The six retrievers over one set of pages.

    A new one builds each retriever on first use; ``save`` builds them all and ``load``
    reads them back without embedding any page.
    """

    def __init__(
        self,
        documents: List[Document],
        embed_model,
        llm,
        chroma_dir: Optional[str] = None,
        previous_dir: Optional[str] = None,
    ):
        """``previous_dir`` is an older saved version, only read, whose vectors and page
        summaries are reused. It must have been built with the same embedding model."""
        if not documents:
            raise ValueError("No documents to index. Scrape a page first.")
        self._init_common(embed_model, llm, chroma_dir or config.CHROMA_DIR)
        self.documents = documents
        self.chunks = self._splitter(config.CHUNK_SIZES[0]).get_nodes_from_documents(documents)
        self._previous = None
        self._saved_summaries = {}
        if previous_dir:
            self._previous = chromadb.PersistentClient(path=str(previous_dir))
            summaries = Path(previous_dir, "summaries.json")
            if summaries.exists():
                self._saved_summaries = json.loads(summaries.read_text(encoding="utf-8"))

    def _init_common(self, embed_model, llm, chroma_dir) -> None:
        self.embed_model = embed_model
        self.llm = llm
        self.chroma_dir = Path(chroma_dir)
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self._retrievers = {}
        self._build_lock = threading.Lock()

    def save(self) -> None:
        """Build all six retrievers and write what ``load`` needs next to the Chroma files."""
        for name in RETRIEVER_NAMES:
            self.retriever(name)
        # vectors are already in Chroma, so they are not duplicated in the JSON file
        chunks = [{**chunk.to_dict(), "embedding": None} for chunk in self.chunks]
        self._write_json("chunks.json", chunks)
        self._auto_merging_storage.persist(persist_dir=str(self.chroma_dir / "auto_merging"))
        index = self._summary_index
        index.storage_context.persist(persist_dir=str(self.chroma_dir / "document_summary"))
        self._write_json("summaries.json", {
            document.doc_id: {
                "hash": text_hash(document.text),
                "summary": index.get_document_summary(document.doc_id),
            }
            for document in self.documents
        })

    def _write_json(self, name: str, data) -> None:
        (self.chroma_dir / name).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, version_dir: str, embed_model, llm) -> "KnowledgeBase":
        """Read a saved version. All retrievers are loaded now, so damage shows up here."""
        folder = Path(version_dir)
        # read first: opening Chroma on a missing folder would silently create it
        data = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        knowledge_base = cls.__new__(cls)
        knowledge_base._init_common(embed_model, llm, folder)
        knowledge_base.chunks = [TextNode.from_dict(item) for item in data]
        knowledge_base._load_retrievers()
        return knowledge_base

    def _stored_index(self, collection: str) -> VectorStoreIndex:
        """Open an existing Chroma collection without writing to it (fails if missing)."""
        store = ChromaVectorStore(chroma_collection=self.client.get_collection(collection))
        return VectorStoreIndex.from_vector_store(store, embed_model=self.embed_model)

    def _load_retrievers(self) -> None:
        self.__dict__["_chunk_index"] = self._stored_index("chunks")  # fills cached_property
        leaves = self._stored_index("leaves")
        auto_merging_storage = StorageContext.from_defaults(
            persist_dir=str(self.chroma_dir / "auto_merging")
        )
        summary_storage = StorageContext.from_defaults(
            persist_dir=str(self.chroma_dir / "document_summary"),
            vector_store=ChromaVectorStore(
                chroma_collection=self.client.get_collection("summaries")
            ),
        )
        summary_index = load_index_from_storage(
            summary_storage, embed_model=self.embed_model, llm=self.llm
        )
        self._retrievers = {
            "vector": self._build_vector(),
            "bm25": self._build_bm25(),
            "hybrid": self._build_hybrid(),
            "auto_merging": self._auto_merging_retriever(leaves, auto_merging_storage),
            "recursive": self._recursive_retriever(self._stored_index("children")),
            "document_summary": self._summary_retriever(summary_index),
        }

    @cached_property
    def _chunk_index(self) -> VectorStoreIndex:
        """Shared by "vector" and "hybrid": rebuilding it would drop the Chroma collection."""
        return self._vector_index(self.chunks, "chunks")

    def retriever(self, name: str) -> BaseRetriever:
        """Return the retriever called ``name`` (one of RETRIEVER_NAMES)."""
        if name not in RETRIEVER_NAMES:
            raise ValueError(f"Unknown retriever '{name}'. Choose from {RETRIEVER_NAMES}")
        if name not in self._retrievers:
            # the chat and the indexing run in separate threads: never build twice at once,
            # because building replaces the Chroma collection
            with self._build_lock:
                if name not in self._retrievers:
                    logger.info("Building '%s' retriever", name)
                    self._retrievers[name] = getattr(self, f"_build_{name}")()
        return self._retrievers[name]

    @staticmethod
    def _splitter(chunk_size: int) -> SentenceSplitter:
        return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=config.CHUNK_OVERLAP)

    def _vector_index(self, nodes, collection: str) -> VectorStoreIndex:
        """Embed ``nodes`` into the Chroma collection ``collection``."""
        reuse_vectors(self._previous, collection, nodes)
        return VectorStoreIndex(
            nodes,
            storage_context=chroma_storage(self.client, collection),
            embed_model=self.embed_model,
        )

    def _build_vector(self) -> BaseRetriever:
        """Semantic similarity over 512-token chunks."""
        return self._chunk_index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)

    def _bm25(self, top_k: int) -> BaseRetriever:
        # No stemming and no stopword list: both are language specific, and pages
        # can be French or English.
        return BM25Retriever.from_defaults(
            nodes=self.chunks,
            similarity_top_k=min(top_k, len(self.chunks)),
            skip_stemming=True,
            language=[],
        )

    def _build_bm25(self) -> BaseRetriever:
        """Keyword matching. Works in memory, so it uses no embeddings."""
        return self._bm25(config.SIMILARITY_TOP_K)

    def _build_hybrid(self) -> BaseRetriever:
        """Semantic and keyword search together, merged by weighted scores."""
        return HybridRetriever(
            self._chunk_index.as_retriever(similarity_top_k=config.HYBRID_CANDIDATES),
            self._bm25(config.HYBRID_CANDIDATES),
            vector_weight=config.HYBRID_VECTOR_WEIGHT,
            top_k=config.SIMILARITY_TOP_K,
        )

    def _build_document_summary(self) -> BaseRetriever:
        """Pick pages by their LLM summary, then return all chunks of those pages."""
        summarizer = CachedSummarizer(
            get_response_synthesizer(llm=self.llm, response_mode="tree_summarize"),
            self._saved_summaries,
            {document.doc_id: text_hash(document.text) for document in self.documents},
        )
        storage = chroma_storage(self.client, "summaries")
        index = DocumentSummaryIndex.from_documents(
            self.documents,
            llm=self.llm,
            embed_model=self.embed_model,
            storage_context=storage,
            transformations=[self._splitter(config.CHUNK_SIZES[0])],
            response_synthesizer=summarizer,
            embed_summaries=False,  # embedded below, so that stored vectors can be reused
        )
        summaries = [
            index.docstore.get_node(summary_id).model_copy()
            for summary_id in index.index_struct.doc_id_to_summary_id.values()
        ]
        reuse_vectors(self._previous, "summaries", summaries)
        vectors = embed_nodes(summaries, self.embed_model)
        for summary in summaries:
            summary.embedding = vectors[summary.node_id]
        storage.vector_store.add(summaries)
        self._summary_index = index  # kept for save()
        return self._summary_retriever(index)

    def _summary_retriever(self, index) -> BaseRetriever:
        return DocumentSummaryIndexEmbeddingRetriever(
            index, similarity_top_k=config.SUMMARY_TOP_K, embed_model=self.embed_model
        )

    def _build_auto_merging(self) -> BaseRetriever:
        """Search small leaf chunks; replace them by their parent when enough siblings match."""
        nodes = HierarchicalNodeParser.from_defaults(
            chunk_sizes=config.CHUNK_SIZES, chunk_overlap=config.CHUNK_OVERLAP
        ).get_nodes_from_documents(self.documents)
        leaves = get_leaf_nodes(nodes)
        storage = chroma_storage(self.client, "leaves")
        storage.docstore.add_documents(nodes)  # parents are looked up here when merging
        reuse_vectors(self._previous, "leaves", leaves)
        index = VectorStoreIndex(leaves, storage_context=storage, embed_model=self.embed_model)
        self._auto_merging_storage = storage  # kept for save()
        return self._auto_merging_retriever(index, storage)

    def _auto_merging_retriever(self, leaf_index, storage) -> BaseRetriever:
        leaf_retriever = leaf_index.as_retriever(similarity_top_k=config.LEAF_TOP_K)
        return AutoMergingRetriever(leaf_retriever, storage)

    def _build_recursive(self) -> BaseRetriever:
        """Search small child chunks, then follow their reference to the full parent chunk."""
        splitter = self._splitter(config.CHUNK_SIZES[-1])
        children = [
            IndexNode(text=text, index_id=parent.node_id)
            for parent in self.chunks
            for text in splitter.split_text(parent.get_content())
        ]
        return self._recursive_retriever(self._vector_index(children, "children"))

    def _recursive_retriever(self, child_index) -> BaseRetriever:
        return RecursiveRetriever(
            "vector",
            retriever_dict={
                "vector": child_index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
            },
            node_dict={parent.node_id: parent for parent in self.chunks},
        )
