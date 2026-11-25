"""Module for interfacing with local LLMs (Ollama) and embeddings (HuggingFace).

Function names keep the original ``watsonx`` naming so callers do not change.
"""

import logging

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.ollama import Ollama

import config

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_EMBEDDING_MODEL_ID = getattr(config, "LOCAL_EMBEDDING_MODEL_ID", "BAAI/bge-m3")


def create_watsonx_embedding() -> BaseEmbedding:
    """Creates a local HuggingFace embedding model for vector representation.

    Returns:
        HuggingFaceEmbedding model.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # heavy import

    return HuggingFaceEmbedding(model_name=LOCAL_EMBEDDING_MODEL_ID, device=config.EMBEDDING_DEVICE)


def create_watsonx_llm(
    temperature: float = 0.0,
    max_new_tokens: int = 500,
    decoding_method: str = "sample",
) -> Ollama:
    """Creates an Ollama LLM for generating responses.

    Args:
        temperature: Randomness of generation (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.
        decoding_method: "greedy" forces temperature 0; anything else uses ``temperature``.

    Returns:
        Ollama LLM using config.LLM_MODEL_ID.
    """
    if decoding_method == "greedy":
        temperature = 0.0
    return Ollama(
        model=config.LLM_MODEL_ID,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        request_timeout=120.0,
        context_window=4096,
        # num_ctx caps the KV cache/compute buffers so it fits a 6 GB GPU
        additional_kwargs={"num_predict": max_new_tokens, "num_ctx": 4096},
    )


def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.

    Args:
        new_model_id: New Ollama model name (e.g. "llama3.2").
    """
    old = config.LLM_MODEL_ID
    config.LLM_MODEL_ID = new_model_id
    logger.info("LLM model changed: %s -> %s", old, new_model_id)
