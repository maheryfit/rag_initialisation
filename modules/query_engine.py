"""Answer questions with a routed (or chosen) retriever and the local LLM."""

from typing import Any, Tuple

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine

from modules.retrievers import KnowledgeBase
from modules.router import route_query
import config

AUTO = "auto"


def answer_user_query(
    knowledge_base: KnowledgeBase, user_query: str, retriever_name: str = AUTO
) -> Tuple[Any, str]:
    """Retrieve context and let the LLM answer.

    Args:
        knowledge_base: Indexed documents.
        user_query: The user's question.
        retriever_name: A retriever name, or "auto" to let the router choose.

    Returns:
        (LlamaIndex response, name of the retriever that was used).
    """
    if retriever_name == AUTO:
        retriever_name = route_query(user_query, knowledge_base.llm)
    engine = RetrieverQueryEngine.from_args(
        knowledge_base.retriever(retriever_name),
        llm=knowledge_base.llm,
        text_qa_template=PromptTemplate(config.USER_QUESTION_TEMPLATE),
    )
    return engine.query(user_query), retriever_name


def source_urls(response: Any) -> list:
    """Unique page URLs behind a response, in retrieval order."""
    urls = (n.node.metadata.get("url") or n.node.ref_doc_id for n in response.source_nodes)
    return list(dict.fromkeys(u for u in urls if u))
