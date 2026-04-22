from __future__ import annotations


def normalize_perplexities(perplexities: list[float]) -> list[float]:
    if not perplexities:
        return []
    low = min(perplexities)
    high = max(perplexities)
    if high <= low:
        return [0.0 for _ in perplexities]
    return [(p - low) / (high - low) for p in perplexities]


def composite_score(
    normalized_perplexity: float,
    embedding_similarity: float,
    perplexity_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> float:
    semantic_risk = embedding_similarity
    score = (perplexity_weight * normalized_perplexity) + (semantic_weight * semantic_risk)
    return max(0.0, min(1.0, score))
