"""
RAG modules package.

This package contains the following modules:
- data_extraction: Functions for scraping web pages into JSON
- web_spider: Scrapy spider (run in a separate process)
- data_processing: Loading the JSON files and preparing Chroma DB
- retrievers: The six LlamaIndex retrievers
- versions: Index version folders and the pointer to the active one
- indexer: Background builds that switch version only after full success
- router: LLM query router that picks a retriever
- llm_interface: Functions for interfacing with local Ollama LLMs and HuggingFace embeddings
- query_engine: Functions for answering questions
"""

from modules.data_extraction import scrape_web_page
from modules.data_processing import load_documents
from modules.llm_interface import create_watsonx_embedding, create_watsonx_llm, change_llm_model
from modules.retrievers import KnowledgeBase, RETRIEVER_NAMES
from modules.router import route_query
from modules.query_engine import answer_user_query
