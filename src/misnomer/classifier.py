from __future__ import annotations


def classify_substitution(embedding_similarity: float, threshold: float = 0.5) -> str:
    if embedding_similarity >= threshold:
        return "SEMANTIC"
    return "OBVIOUS"
