"""
Embedding model wrapper for SMS Spam Detector.

Handles loading and caching of the SentenceTransformer model,
and computing text embeddings.
"""

import numpy as np
from typing import List, Union
import streamlit as st
from sentence_transformers import SentenceTransformer


# Default embedding model - lightweight and effective
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Load the SentenceTransformer model with caching.

    Uses Streamlit's cache_resource to ensure the model is loaded only once
    per session, improving performance significantly.

    Args:
        model_name: Name of the SentenceTransformer model to load.

    Returns:
        Loaded SentenceTransformer model.
    """
    return SentenceTransformer(model_name)


def compute_embeddings(
    texts: Union[List[str], str],
    model: SentenceTransformer = None,
    model_name: str = DEFAULT_MODEL_NAME,
    show_progress: bool = False,
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute embeddings for given texts.

    Args:
        texts: Single text or list of texts to embed.
        model: Pre-loaded SentenceTransformer model. If None, will load.
        model_name: Model name to load if model is None.
        show_progress: Whether to show a progress bar.
        batch_size: Batch size for encoding.

    Returns:
        Numpy array of embeddings. Shape: (n_texts, embedding_dim)
    """
    if model is None:
        model = load_embedding_model(model_name)

    # Handle single text input
    if isinstance(texts, str):
        texts = [texts]

    # Compute embeddings
    embeddings = model.encode(
        texts,
        show_progress_bar=show_progress,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalise for cosine similarity
    )

    return embeddings


def get_embedding_dimension(model_name: str = DEFAULT_MODEL_NAME) -> int:
    """
    Get the embedding dimension for a given model.

    Args:
        model_name: Name of the SentenceTransformer model.

    Returns:
        Embedding dimension size.
    """
    model = load_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()


def get_model_info(model_name: str = DEFAULT_MODEL_NAME) -> dict:
    """
    Get information about the embedding model.

    Args:
        model_name: Name of the SentenceTransformer model.

    Returns:
        Dictionary with model information.
    """
    model = load_embedding_model(model_name)

    return {
        "model_name": model_name,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_sequence_length": model.max_seq_length,
    }
