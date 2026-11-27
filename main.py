"""Command-line version: scrape a page, index it in Chroma, and chat with it."""

import sys
import logging
import argparse
from functools import lru_cache

from modules.data_extraction import scrape_web_page
from modules.data_processing import load_documents
from modules.indexer import Indexer
from modules.llm_interface import change_llm_model, create_watsonx_embedding, create_watsonx_llm
from modules.query_engine import answer_user_query, source_urls, AUTO
from modules.retrievers import RETRIEVER_NAMES
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)


def chatbot_interface(knowledge_base, retriever_name):
    """Ask questions until the user types 'exit', 'quit' or 'bye'."""
    print("\nPose ta question / Ask a question ('exit' to quit).")
    while True:
        query = input("\n> ").strip()
        if query.lower() in {"exit", "quit", "bye"}:
            return
        if not query:
            continue
        response, used = answer_user_query(knowledge_base, query, retriever_name)
        print(f"\n{response}\n\n[retriever: {used}]")
        for url in source_urls(response):
            print(f"  - {url}")


def main():
    parser = argparse.ArgumentParser(description="Web page RAG")
    parser.add_argument("--url", help="Web page to scrape (omit to use the saved JSON files)")
    parser.add_argument("--retriever", choices=[AUTO] + RETRIEVER_NAMES, default=AUTO)
    parser.add_argument("--model", help='Ollama model to use (e.g., "llama3.2")')
    args = parser.parse_args()

    if args.model:
        change_llm_model(args.model)
    if args.url and not scrape_web_page(args.url):
        sys.exit("Nothing scraped. See the log above.")

    indexer = Indexer(
        config.CHROMA_DIR, load_documents,
        lru_cache(maxsize=1)(create_watsonx_embedding), create_watsonx_llm,
    )
    indexer.load()
    if args.url or indexer.knowledge_base is None:
        indexer.request()
        indexer.wait(None)
    # after a failed build this is the previous version, if there is one
    print(indexer.status()["message"])
    if indexer.knowledge_base is None:
        sys.exit(1)
    chatbot_interface(indexer.knowledge_base, args.retriever)


if __name__ == "__main__":
    main()
