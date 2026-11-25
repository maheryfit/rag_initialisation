"""Load scraped JSON files and prepare Chroma DB storage."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from llama_index.core import Document, StorageContext
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
    # ponytail: full rebuild on every indexing run, single user. Use LlamaIndex's
    # IngestionPipeline with a persisted docstore when re-embedding becomes too slow.
    client.get_or_create_collection(collection_name)
    client.delete_collection(collection_name)
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    return StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
