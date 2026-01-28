"""
Explanation utilities for SMS Spam Detector.

Provides functions for explaining predictions through token overlap analysis.
"""

import re
from typing import List, Set, Tuple
from collections import Counter


# Common stop words to exclude from overlap analysis
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "its", "our", "their", "what", "which", "who", "whom", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "also", "now", "here", "there", "then",
    "if", "else", "about", "into", "over", "after", "before", "between",
    "through", "during", "under", "again", "once", "any", "up", "down",
    "out", "off", "get", "got", "u", "ur", "r", "im", "dont", "wont",
    "cant", "didnt", "its", "youre", "thats", "ill", "ive", "youve"
}


def tokenise(text: str) -> List[str]:
    """
    Simple tokenisation of text into words.

    Args:
        text: Input text string.

    Returns:
        List of lowercase tokens.
    """
    # Convert to lowercase and extract word-like tokens
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return tokens


def get_meaningful_tokens(text: str, min_length: int = 2) -> Set[str]:
    """
    Extract meaningful tokens from text, excluding stop words.

    Args:
        text: Input text string.
        min_length: Minimum token length to include.

    Returns:
        Set of meaningful tokens.
    """
    tokens = tokenise(text)
    meaningful = {
        t for t in tokens
        if len(t) >= min_length and t not in STOP_WORDS
    }
    return meaningful


def compute_token_overlap(text1: str, text2: str) -> Tuple[Set[str], float]:
    """
    Compute the token overlap between two texts.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Tuple of (common tokens, Jaccard similarity score).
    """
    tokens1 = get_meaningful_tokens(text1)
    tokens2 = get_meaningful_tokens(text2)

    if not tokens1 or not tokens2:
        return set(), 0.0

    common = tokens1 & tokens2
    union = tokens1 | tokens2

    jaccard = len(common) / len(union) if union else 0.0

    return common, jaccard


def explain_prediction(input_text: str, neighbours: List[dict]) -> List[dict]:
    """
    Generate explanations for a prediction based on neighbour analysis.

    Args:
        input_text: The input text that was classified.
        neighbours: List of neighbour dictionaries from prediction.

    Returns:
        List of neighbour explanations with token overlap info.
    """
    explanations = []

    for neighbour in neighbours:
        common_tokens, jaccard = compute_token_overlap(
            input_text,
            neighbour["text"]
        )

        explanations.append({
            "rank": neighbour["rank"],
            "label": neighbour["label"],
            "similarity": neighbour["similarity"],
            "text": neighbour["text"],
            "text_snippet": truncate_text(neighbour["text"], max_length=100),
            "common_tokens": sorted(common_tokens),
            "token_overlap_score": jaccard,
            "num_common_tokens": len(common_tokens)
        })

    return explanations


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Args:
        text: Input text.
        max_length: Maximum length before truncation.

    Returns:
        Truncated text with ellipsis if needed.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3].rsplit(' ', 1)[0] + "..."


def get_common_spam_indicators(neighbours: List[dict]) -> List[str]:
    """
    Identify common tokens among spam neighbours.

    Args:
        neighbours: List of neighbour dictionaries.

    Returns:
        List of common tokens found in spam neighbours.
    """
    spam_neighbours = [n for n in neighbours if n["label"] == "spam"]

    if not spam_neighbours:
        return []

    # Count token frequencies across spam neighbours
    token_counts = Counter()
    for n in spam_neighbours:
        tokens = get_meaningful_tokens(n["text"])
        token_counts.update(tokens)

    # Return tokens that appear in multiple spam neighbours
    common = [
        token for token, count in token_counts.most_common(10)
        if count >= min(2, len(spam_neighbours))
    ]

    return common


def format_explanation_summary(prediction_result: dict) -> str:
    """
    Format a human-readable explanation summary.

    Args:
        prediction_result: Full prediction result dictionary.

    Returns:
        Formatted explanation string.
    """
    pred = prediction_result["prediction"].upper()
    conf = prediction_result["confidence"] * 100
    spam_votes = prediction_result["spam_votes"]
    ham_votes = prediction_result["ham_votes"]
    k = spam_votes + ham_votes

    summary_lines = [
        f"**Prediction:** {pred}",
        f"**Confidence:** {conf:.1f}%",
        f"**Neighbour votes:** {spam_votes} spam, {ham_votes} ham (out of {k})",
        "",
        "**Reasoning:**"
    ]

    if pred == "SPAM":
        summary_lines.append(
            f"The message is similar to {spam_votes} known spam messages in the training data."
        )
    else:
        summary_lines.append(
            f"The message is similar to {ham_votes} known legitimate (ham) messages in the training data."
        )

    return "\n".join(summary_lines)
