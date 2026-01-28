"""
Model training and prediction module for SMS Spam Detector.

Handles KNN classifier training, prediction, and evaluation.
"""

import os
import re
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .embedder import compute_embeddings, load_embedding_model, DEFAULT_MODEL_NAME


# Directory for saving artefacts
ARTEFACTS_DIR = Path(__file__).parent.parent / "artefacts"


def clean_text(text: str) -> str:
    """
    Lightly clean text for embedding.

    Only performs minimal cleaning to preserve semantic content:
    - Strip leading/trailing whitespace
    - Normalise multiple whitespace to single space

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    # Strip and normalise whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data by splitting into train and test sets.

    Args:
        df: Full dataset DataFrame.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    # Clean text
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)

    # Stratified split to maintain class balance
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_model(
    train_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    n_neighbors: int = 5,
    progress_callback=None
) -> Tuple[KNeighborsClassifier, np.ndarray]:
    """
    Train KNN classifier on training data.

    Args:
        train_df: Training DataFrame with 'text_clean' and 'label' columns.
        model_name: SentenceTransformer model name for embeddings.
        n_neighbors: Default number of neighbours for KNN.
        progress_callback: Optional callback for progress updates.

    Returns:
        Tuple of (trained KNN classifier, training embeddings).
    """
    if progress_callback:
        progress_callback("Loading embedding model...")

    embedding_model = load_embedding_model(model_name)

    if progress_callback:
        progress_callback("Computing embeddings for training data...")

    # Compute embeddings for training texts
    train_embeddings = compute_embeddings(
        train_df["text_clean"].tolist(),
        model=embedding_model,
        show_progress=False
    )

    if progress_callback:
        progress_callback("Training KNN classifier...")

    # Train KNN with cosine distance and distance weighting
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="cosine",
        weights="distance",
        algorithm="brute"  # Required for cosine metric
    )

    knn.fit(train_embeddings, train_df["label"].values)

    return knn, train_embeddings


def predict_single(
    text: str,
    knn: KNeighborsClassifier,
    train_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    k: int = 5
) -> Dict[str, Any]:
    """
    Predict label for a single text with explanation.

    Args:
        text: Input text to classify.
        knn: Trained KNN classifier.
        train_embeddings: Training set embeddings.
        train_df: Training DataFrame.
        model_name: Embedding model name.
        k: Number of neighbours to consider.

    Returns:
        Dictionary with prediction results and explanation.
    """
    # Clean and embed the input text
    text_clean = clean_text(text)
    embedding = compute_embeddings(text_clean, model_name=model_name)

    # Get k nearest neighbours
    distances, indices = knn.kneighbors(embedding, n_neighbors=k)
    distances = distances[0]  # Flatten
    indices = indices[0]

    # Convert cosine distance to similarity (1 - distance)
    similarities = 1 - distances

    # Get neighbour information
    neighbours = []
    for idx, (train_idx, sim) in enumerate(zip(indices, similarities)):
        neighbours.append({
            "rank": idx + 1,
            "text": train_df.iloc[train_idx]["text"],
            "label": train_df.iloc[train_idx]["label"],
            "similarity": float(sim)
        })

    # Compute weighted vote
    spam_weight = sum(sim for n, sim in zip(neighbours, similarities) if n["label"] == "spam")
    ham_weight = sum(sim for n, sim in zip(neighbours, similarities) if n["label"] == "ham")
    total_weight = spam_weight + ham_weight

    # Prediction based on weighted majority
    if spam_weight > ham_weight:
        prediction = "spam"
        confidence = spam_weight / total_weight if total_weight > 0 else 0.5
    else:
        prediction = "ham"
        confidence = ham_weight / total_weight if total_weight > 0 else 0.5

    # Simple vote counts
    spam_votes = sum(1 for n in neighbours if n["label"] == "spam")
    ham_votes = sum(1 for n in neighbours if n["label"] == "ham")

    return {
        "input_text": text,
        "input_text_clean": text_clean,
        "prediction": prediction,
        "confidence": float(confidence),
        "spam_votes": spam_votes,
        "ham_votes": ham_votes,
        "spam_weight": float(spam_weight),
        "ham_weight": float(ham_weight),
        "neighbours": neighbours
    }


def predict_batch(
    texts: List[str],
    knn: KNeighborsClassifier,
    train_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    k: int = 5,
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Predict labels for multiple texts.

    Args:
        texts: List of input texts to classify.
        knn: Trained KNN classifier.
        train_embeddings: Training set embeddings.
        train_df: Training DataFrame.
        model_name: Embedding model name.
        k: Number of neighbours to consider.
        progress_callback: Optional callback for progress updates.

    Returns:
        List of prediction results.
    """
    results = []
    total = len(texts)

    for i, text in enumerate(texts):
        if progress_callback:
            progress_callback(i / total)

        result = predict_single(
            text, knn, train_embeddings, train_df, model_name, k
        )
        results.append(result)

    return results


def evaluate_model(
    test_df: pd.DataFrame,
    knn: KNeighborsClassifier,
    model_name: str = DEFAULT_MODEL_NAME
) -> Dict[str, float]:
    """
    Evaluate model performance on test set.

    Args:
        test_df: Test DataFrame with 'text_clean' and 'label' columns.
        knn: Trained KNN classifier.
        model_name: Embedding model name.

    Returns:
        Dictionary with evaluation metrics.
    """
    # Compute test embeddings
    test_embeddings = compute_embeddings(
        test_df["text_clean"].tolist(),
        model_name=model_name,
        show_progress=False
    )

    # Predict
    y_pred = knn.predict(test_embeddings)
    y_true = test_df["label"].values

    # Compute metrics
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="spam")),
        "recall": float(recall_score(y_true, y_pred, pos_label="spam")),
        "f1": float(f1_score(y_true, y_pred, pos_label="spam"))
    }


def save_artefacts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    knn: KNeighborsClassifier,
    metrics: Dict[str, float],
    file_hash: str,
    model_name: str = DEFAULT_MODEL_NAME
) -> None:
    """
    Save trained artefacts to disk for later reuse.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        train_embeddings: Training embeddings.
        knn: Trained KNN classifier.
        metrics: Evaluation metrics.
        file_hash: Hash of the source data file.
        model_name: Embedding model name used.
    """
    # Ensure artefacts directory exists
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save components
    joblib.dump(train_embeddings, ARTEFACTS_DIR / "train_embeddings.joblib")
    joblib.dump(knn, ARTEFACTS_DIR / "knn_model.joblib")
    train_df.to_pickle(ARTEFACTS_DIR / "train_df.pkl")
    test_df.to_pickle(ARTEFACTS_DIR / "test_df.pkl")

    # Save metadata
    metadata = {
        "file_hash": file_hash,
        "model_name": model_name,
        "metrics": metrics,
        "train_size": len(train_df),
        "test_size": len(test_df)
    }
    with open(ARTEFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_artefacts(
    file_hash: str,
    model_name: str = DEFAULT_MODEL_NAME
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, KNeighborsClassifier, Dict[str, float]]]:
    """
    Load saved artefacts if they exist and match the current data.

    Args:
        file_hash: Hash of the current data file.
        model_name: Expected embedding model name.

    Returns:
        Tuple of (train_df, test_df, train_embeddings, knn, metrics) or None if invalid.
    """
    metadata_path = ARTEFACTS_DIR / "metadata.json"

    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Check if artefacts match current data and model
        if metadata.get("file_hash") != file_hash:
            return None
        if metadata.get("model_name") != model_name:
            return None

        # Load all components
        train_embeddings = joblib.load(ARTEFACTS_DIR / "train_embeddings.joblib")
        knn = joblib.load(ARTEFACTS_DIR / "knn_model.joblib")
        train_df = pd.read_pickle(ARTEFACTS_DIR / "train_df.pkl")
        test_df = pd.read_pickle(ARTEFACTS_DIR / "test_df.pkl")
        metrics = metadata.get("metrics", {})

        return train_df, test_df, train_embeddings, knn, metrics

    except Exception:
        return None


def check_artefacts_exist() -> bool:
    """Check if artefacts directory has saved models."""
    return (ARTEFACTS_DIR / "metadata.json").exists()
