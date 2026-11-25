"""Configuration settings for the Icebreaker Bot."""

# Model settings
LLM_MODEL_ID = "llama3.2"  # Ollama model name
OLLAMA_BASE_URL = "http://localhost:11434"
LOCAL_EMBEDDING_MODEL_ID = "BAAI/bge-m3"  # multilingual (French, English, ...)
EMBEDDING_DEVICE = "cpu"  # keeps the GPU free for llama3.2; use "cuda" if VRAM allows

# Folder where scraped web pages are saved as JSON
SCRAPED_DATA_DIR = "scraped_data"

# Query settings
SIMILARITY_TOP_K = 5
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
MIN_NEW_TOKENS = 1
TOP_K = 50
TOP_P = 1

# Node settings, in tokens (bge-m3 accepts up to 8192).
CHUNK_SIZES = [512, 256, 128]  # parent -> leaf
CHUNK_OVERLAP = 20

# Retriever settings
LEAF_TOP_K = 12  # leaves searched by auto merging; higher means more merges
SUMMARY_TOP_K = 1  # pages selected by the document summary retriever
HYBRID_VECTOR_WEIGHT = 0.7  # BM25 gets the rest (0.3)
HYBRID_CANDIDATES = 10  # results taken from each side before merging

# Chroma DB folder
CHROMA_DIR = "chroma_db"

# LLM prompt templates
USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, in the same language as the question (French or English), using only the information provided in the context. If the answer is not available in the context, say that you don't know because the scraped pages do not contain it.
"""
