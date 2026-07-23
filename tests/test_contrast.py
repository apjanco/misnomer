"""Tests for the surprisal-contrast signal (scorer_version >= 1.1).

Two groups:

* Pure-function tests for ``contrast_to_override_risk`` — deterministic, no
  model required.
* Model-backed tests for ``LMScorer.first_token_surprisal`` and the end-to-end
  substitution path — skipped when torch/sentence-transformers are unavailable.
"""

import math
import importlib.util

import pytest

from misnomer.composite import contrast_to_override_risk
from misnomer.config import ScorerConfig


# ---------------------------------------------------------------------------
# Pure-function: contrast_to_override_risk
# ---------------------------------------------------------------------------

def test_override_risk_three_regimes() -> None:
    # Δ ≪ 0: predicted far less surprising than gold → override → risk → 1
    assert contrast_to_override_risk(-3.594) > 0.9
    # Δ = 0: model cannot separate → no evidence → 0.5
    assert contrast_to_override_risk(0.0) == pytest.approx(0.5)
    # Δ > 0: predicted more surprising → character noise → risk → 0
    assert contrast_to_override_risk(5.359) < 0.1


def test_override_risk_softmax_identity() -> None:
    """At T=1, σ(−Δ) == P(pred)/(P(pred)+P(gold)) with S = −log P."""
    s_gold, s_pred = 4.730, 1.135
    delta = s_pred - s_gold
    p_gold, p_pred = math.exp(-s_gold), math.exp(-s_pred)
    assert contrast_to_override_risk(delta) == pytest.approx(
        p_pred / (p_pred + p_gold)
    )


def test_override_risk_monotonic_decreasing_in_delta() -> None:
    xs = [-5, -2, -0.5, 0, 0.5, 2, 5]
    risks = [contrast_to_override_risk(x) for x in xs]
    assert all(a > b for a, b in zip(risks, risks[1:]))


def test_override_risk_temperature() -> None:
    d = -3.594
    sharp = contrast_to_override_risk(d, temperature=0.5)
    base = contrast_to_override_risk(d, temperature=1.0)
    flat = contrast_to_override_risk(d, temperature=2.0)
    # lower T pushes toward the extreme (1); higher T flattens toward 0.5
    assert sharp > base > flat > 0.5


def test_override_risk_positive_temperature_guard() -> None:
    with pytest.raises(ValueError):
        contrast_to_override_risk(1.0, temperature=0.0)
    with pytest.raises(ValueError):
        contrast_to_override_risk(1.0, temperature=-1.0)


def test_override_risk_in_unit_interval() -> None:
    for d in (-50, -3, 0, 3, 50):
        r = contrast_to_override_risk(d)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Model-backed: first_token_surprisal + end-to-end substitution path
# ---------------------------------------------------------------------------

_HAS_MODEL = importlib.util.find_spec("torch") and importlib.util.find_spec(
    "sentence_transformers"
)


@pytest.fixture(scope="module")
def lm():
    from misnomer.models.lm import LMScorer

    return LMScorer(ScorerConfig(local_files_only=False, allow_download=True))


@pytest.mark.skipif(not _HAS_MODEL, reason="requires torch and sentence-transformers")
def test_fluent_continuation_ranks_below_surprising(lm) -> None:
    ctx = "she poured the coffee into the"
    s_fluent = lm.first_token_surprisal(ctx, "cup")
    s_surprising = lm.first_token_surprisal(ctx, "onto")
    assert s_fluent is not None and s_surprising is not None
    assert s_fluent < s_surprising


@pytest.mark.skipif(not _HAS_MODEL, reason="requires torch and sentence-transformers")
def test_position_zero_returns_none(lm) -> None:
    # empty left-context: the word starts the sequence, no predecessor to
    # condition on, so there is no contextual surprisal.
    assert lm.first_token_surprisal("", "steed") is None
    # empty / whitespace word
    assert lm.first_token_surprisal("the cavalry charged on his", "") is None


@pytest.mark.skipif(not _HAS_MODEL, reason="requires torch and sentence-transformers")
def test_rarity_cancels_when_word_read_correctly(lm) -> None:
    """A rare word read *correctly* has pred == gold, so the contrast is
    exactly 0 no matter how surprising the word is — the intrinsic rarity of
    the slot cancels in the difference. This is why the contrast, unlike raw
    gold-only perplexity, does not flag rare-but-correct words."""
    ctx = "the letter was addressed to Mr"
    rare = "Fotheringay"
    s = lm.first_token_surprisal(ctx, rare)
    assert s is not None and s > 3.0, "the rare word should be surprising on its own"
    contrast = s - s  # pred == gold
    assert contrast == 0.0
    assert contrast_to_override_risk(contrast) == pytest.approx(0.5)


@pytest.mark.skipif(not _HAS_MODEL, reason="requires torch and sentence-transformers")
def test_canonical_override_negative_contrast(lm) -> None:
    """Manuscript 'steed' misread as the common 'horse': predicted word is less
    surprising than gold → negative contrast → override_risk > 0.5."""
    from misnomer import score

    cfg = ScorerConfig(local_files_only=False, allow_download=True)
    r = score("The horse stood in the field", "The steed stood in the field", config=cfg)
    sub = next(w for w in r.word_scores if w.alignment_type == "SUBSTITUTION")
    assert sub.surprisal_gold is not None and sub.surprisal_pred is not None
    assert sub.surprisal_contrast < 0
    assert sub.override_risk > 0.5


@pytest.mark.skipif(not _HAS_MODEL, reason="requires torch and sentence-transformers")
def test_less_fluent_char_corruption_positive_contrast(lm) -> None:
    """A character corruption of the gold word that reads as a less-fluent real
    word (horse -> hoarse) → positive contrast → override_risk < 0.5."""
    from misnomer import score

    cfg = ScorerConfig(local_files_only=False, allow_download=True)
    r = score("The hoarse stood in the field", "The horse stood in the field", config=cfg)
    sub = next(w for w in r.word_scores if w.alignment_type == "SUBSTITUTION")
    assert sub.surprisal_contrast is not None and sub.surprisal_contrast > 0
    assert sub.override_risk < 0.5


@pytest.mark.skipif(not _HAS_MODEL, reason="requires torch and sentence-transformers")
def test_contrast_disabled_preserves_legacy_fields(lm) -> None:
    """With use_surprisal_contrast=False the new fields stay None and the
    composite falls back to the scorer-1.0 gold-only perplexity leg."""
    from misnomer import score

    cfg = ScorerConfig(
        local_files_only=False, allow_download=True, use_surprisal_contrast=False
    )
    r = score("The horse stood in the field", "The steed stood in the field", config=cfg)
    sub = next(w for w in r.word_scores if w.alignment_type == "SUBSTITUTION")
    assert sub.surprisal_contrast is None
    assert sub.override_risk is None
