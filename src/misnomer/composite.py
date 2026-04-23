from __future__ import annotations


def normalize_perplexities(perplexities: list[float]) -> list[float]:
    if not perplexities:
        return []
    low = min(perplexities)
    high = max(perplexities)
    if high <= low:
        # Degenerate range: all values identical; use neutral midpoint rather than 0
        # so the perplexity signal doesn't silently vanish from composite scores.
        return [0.5 for _ in perplexities]
    return [(p - low) / (high - low) for p in perplexities]


def composite_score(
    normalized_perplexity: float,
    embedding_similarity: float | None,
    perplexity_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> float:
    if embedding_similarity is None:
        # Embedder unavailable: derive score from perplexity alone.
        return max(0.0, min(1.0, normalized_perplexity))
    semantic_risk = embedding_similarity
    score = (perplexity_weight * normalized_perplexity) + (semantic_weight * semantic_risk)
    return max(0.0, min(1.0, score))
