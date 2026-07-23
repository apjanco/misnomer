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


def contrast_to_override_risk(contrast: float, temperature: float = 1.0) -> float:
    """Map a signed first-token surprisal *contrast* to a [0, 1] override risk.

    The contrast is ``Δ = S(pred | c) − S(gold | c)`` in nats, where ``S`` is
    ``−log P`` under the reference LM: how much *more* surprising the predicted
    (OCR'd) word is than the gold word in the same context.

    The mapping is the logistic ``σ(−Δ / T)``. With ``temperature == 1`` this is
    algebraically the reference model's *relative* preference for the predicted
    word over gold::

        σ(−Δ) = 1 / (1 + e^Δ)
              = P(pred | c) / (P(pred | c) + P(gold | c))

    i.e. a two-way softmax between the two candidates. Interpreted as risk that
    the substitution is a *language-prior override*:

    - ``Δ ≪ 0`` (predicted word far *less* surprising than gold — the OCR moved
      the text toward what the LM expects): risk → 1. This is the override
      signature.
    - ``Δ ≈ 0`` (the LM cannot separate the two candidates, e.g. an
      unpredictable name slot, or a correctly-read rare word whose rarity
      cancels): risk ≈ 0.5 — no evidence either way.
    - ``Δ > 0`` (predicted word *more* surprising — the change made the text
      less fluent, the signature of mechanical character corruption): risk → 0.

    ``temperature`` controls sharpness: ``T < 1`` pushes risks toward 0/1,
    ``T > 1`` flattens them toward 0.5. Must be positive.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    # σ(−Δ/T); computed in the numerically stable branchless form.
    z = -contrast / temperature
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


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
