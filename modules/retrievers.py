"""The six LlamaIndex retrievers, built over the scraped documents and Chroma DB."""

import logging
import threading
from functools import cached_property
from typing import List, Optional

import chromadb
from llama_index.core import (
    Document,
    DocumentSummaryIndex,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.indices.document_summary import DocumentSummaryIndexEmbeddingRetriever
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever, RecursiveRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever

from modules.data_processing import chroma_storage
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


class KnowledgeBase:
    """Builds each retriever on first use and keeps it for later queries."""

    def __init__(self, documents: List[Document], embed_model, llm, chroma_dir: Optional[str] = None):
        if not documents:
            raise ValueError("No documents to index. Scrape a page first.")
        self.documents = documents
        self.embed_model = embed_model
        self.llm = llm
        self.client = chromadb.PersistentClient(path=chroma_dir or config.CHROMA_DIR)
        self.chunks = self._splitter(config.CHUNK_SIZES[0]).get_nodes_from_documents(documents)
        self._retrievers = {}
        self._build_lock = threading.Lock()

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
        index = DocumentSummaryIndex.from_documents(
            self.documents,
            llm=self.llm,
            embed_model=self.embed_model,
            storage_context=chroma_storage(self.client, "summaries"),
            transformations=[self._splitter(config.CHUNK_SIZES[0])],
            response_synthesizer=get_response_synthesizer(llm=self.llm, response_mode="tree_summarize"),
        )
        return DocumentSummaryIndexEmbeddingRetriever(
            index, similarity_top_k=config.SUMMARY_TOP_K, embed_model=self.embed_model
        )

    def _build_auto_merging(self) -> BaseRetriever:
        """Search small leaf chunks; replace them by their parent when enough siblings match."""
        nodes = HierarchicalNodeParser.from_defaults(
            chunk_sizes=config.CHUNK_SIZES, chunk_overlap=config.CHUNK_OVERLAP
        ).get_nodes_from_documents(self.documents)
        storage = chroma_storage(self.client, "leaves")
        storage.docstore.add_documents(nodes)  # parents are looked up here when merging
        index = VectorStoreIndex(
            get_leaf_nodes(nodes), storage_context=storage, embed_model=self.embed_model
        )
        return AutoMergingRetriever(index.as_retriever(similarity_top_k=config.LEAF_TOP_K), storage)

    def _build_recursive(self) -> BaseRetriever:
        """Search small child chunks, then follow their reference to the full parent chunk."""
        splitter = self._splitter(config.CHUNK_SIZES[-1])
        children = [
            IndexNode(text=text, index_id=parent.node_id)
            for parent in self.chunks
            for text in splitter.split_text(parent.get_content())
        ]
        index = self._vector_index(children, "children")
        return RecursiveRetriever(
            "vector",
            retriever_dict={"vector": index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)},
            node_dict={parent.node_id: parent for parent in self.chunks},
        )
