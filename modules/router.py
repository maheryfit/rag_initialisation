"""Query router: the LLM picks the retriever that fits the question."""

import logging

from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import ToolMetadata

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVER = "vector"

# The router reads these descriptions to decide. Edit them to tune the routing.
ROUTES = {
    "vector": "General questions about meaning or concepts, even when the wording differs "
              "from the page. Default choice. (question générale sur le sens)",
    "bm25": "Questions about an exact keyword, proper name, quoted phrase, number, code or "
            "identifier. (mot-clé exact, nom propre, citation, référence)",
    "document_summary": "Questions asking for a summary, overview or main topic of a whole page, "
                        "or which page covers a subject. (résumé, vue d'ensemble, sujet d'une page)",
    "auto_merging": "Questions needing a long, detailed explanation spread over several "
                    "neighbouring passages. (explication détaillée sur plusieurs passages)",
    "recursive": "Questions about one precise fact that also needs its surrounding context. "
                 "(fait précis avec son contexte)",
    "hybrid": "Questions mixing a general meaning with specific names or technical terms, "
              "such as a natural language question about a named thing. "
              "(question mêlant sens général et termes précis)",
}


def route_query(query: str, llm, selector=None) -> str:
    """Return the retriever name chosen for ``query``; fall back to DEFAULT_RETRIEVER."""
    names = list(ROUTES)
    try:
        # The selector is used on its own (not RouterRetriever) so that only the chosen
        # retriever gets built; building document_summary costs many LLM calls.
        selector = selector or LLMSingleSelector.from_defaults(llm=llm)
        result = selector.select(
            [ToolMetadata(name=name, description=ROUTES[name]) for name in names], query
        )
        index = result.selections[0].index
        if not 0 <= index < len(names):
            raise IndexError(f"choice {index} out of range")
    except Exception as e:  # small local models sometimes return unparseable output
        logger.warning("Routing failed (%s); using '%s'", e, DEFAULT_RETRIEVER)
        return DEFAULT_RETRIEVER
    logger.info("Router chose '%s': %s", names[index], result.selections[0].reason)
    return names[index]
