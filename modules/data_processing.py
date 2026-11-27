"""Load scraped JSON files and prepare Chroma DB storage."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import chromadb
from chromadb.errors import NotFoundError
from llama_index.core import Document, StorageContext
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

logger = logging.getLogger(__name__)


def load_documents(data_dir: Optional[str] = None) -> List[Document]:
    """Turn every scraped JSON file into one LlamaIndex document per URL."""
    documents = {}
    for path in sorted(Path(data_dir or config.SCRAPED_DATA_DIR).glob("*.json")):
        try:
            pages = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("Skipping unreadable file %s: %s", path, e)
            continue
        for page in pages:
            if not page.get("text"):
                continue
            # "text" already contains the headings, so they are not added twice.
            documents[page["url"]] = Document(
                text=f"{page.get('title', '')}\n{page.get('description', '')}\n{page['text']}".strip(),
                doc_id=page["url"],
                metadata={"url": page["url"], "title": page.get("title", "")},
                excluded_embed_metadata_keys=["url", "title"],
                excluded_llm_metadata_keys=["url", "title"],
            )
    logger.info("Loaded %d documents", len(documents))
    return list(documents.values())


def chroma_storage(client: chromadb.ClientAPI, collection_name: str) -> StorageContext:
    """Storage context backed by an empty Chroma collection (any previous one is replaced)."""
    client.get_or_create_collection(collection_name)
    client.delete_collection(collection_name)
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    return StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))


def reuse_vectors(
    previous_client: Optional[chromadb.ClientAPI],
    collection_name: str,
    nodes: Sequence[BaseNode],
) -> None:
    """Give back its stored vector to each node whose text the previous version already has.

    LlamaIndex only embeds nodes without a vector. The caller must make sure the previous
    version used the same embedding model.
    """
    if previous_client is None or not nodes:
        return
    try:
        collection = previous_client.get_collection(collection_name)
    except NotFoundError:
        return
    stored = collection.get(include=["embeddings", "documents"])
    vectors = dict(zip(stored["documents"], stored["embeddings"]))
    reused = 0
    for node in nodes:
        # matched by text, not ID: the splitter gives new random IDs on every run
        vector = vectors.get(node.get_content(metadata_mode=MetadataMode.EMBED))
        if vector is not None:
            node.embedding = [float(x) for x in vector]
            reused += 1
    logger.info("'%s': reused %d/%d vectors", collection_name, reused, len(nodes))
