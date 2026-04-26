from __future__ import annotations

import math

# Absolute ceiling for log-scale perplexity normalization.
# exp(13.8) ≈ 1_000_000 — perplexities above this are treated as maximally surprising.
_LOG_PPL_MAX = math.log(1_000_000)


def normalize_perplexities(perplexities: list[float]) -> list[float]:
    """Map raw perplexities to [0, 1] using an absolute log scale.

    Each value is normalized independently against a fixed ceiling so that:
    - Single-substitution documents retain a meaningful signal (no more forced 0.5).
    - Scores are comparable across documents.
    - Very high-perplexity (completely surprising) words approach 1.0.
    """
    if not perplexities:
        return []
    return [
        min(1.0, max(0.0, math.log(max(1.0, p)) / _LOG_PPL_MAX))
        for p in perplexities
    ]


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
