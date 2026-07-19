"""Character-anchored word links: unit tests on WordAnchors and end-to-end
segmentation behavior through score(). The artifact classes these tests
encode were all found in the wild (IAM clitic spacing, hyphen spacing,
compound fragments) — see the audit notes."""

from __future__ import annotations

import pytest

from misnomer import score
from misnomer.anchor import WordAnchors
from misnomer.config import ScorerConfig


# ---- unit level -------------------------------------------------------------

def test_join_two_gt_words_one_pred_word():
    # GT convention spaces the clitic; the model writes the surface form.
    a = WordAnchors("It'll be here", "It 'll be here")
    # pred token 0 = "It'll" links strongly into gt tokens 0 ("It") and 1 ("ll")
    confidence, segmentation = a.describe(0, 0)
    assert segmentation == "join"


def test_split_one_gt_word_two_pred_words():
    a = WordAnchors("re - established here", "re-established here")
    # gt token 0 = "re-established"; pred tokens "re" and "established"
    confidence, segmentation = a.describe(0, 0)
    assert segmentation == "split"


def test_fragment_dropped_morpheme():
    a = WordAnchors("the net film", "the seine-net film")
    confidence, segmentation = a.describe(1, 1)
    assert segmentation == "fragment"


def test_ordinary_substitution_is_not_segmentation():
    a = WordAnchors("The parable of the vineyard", "The parable of the vinegar")
    confidence, segmentation = a.describe(4, 4)
    assert segmentation is None
    assert 0.4 < confidence < 0.9  # 'vine' + shared tail chars anchor it


def test_confidence_low_for_positional_pairing():
    # No shared ink: alignment by position only.
    a = WordAnchors("zzzz here", "moon here")
    confidence, segmentation = a.describe(0, 0)
    assert confidence == 0.0
    assert segmentation is None


def test_identical_words_full_confidence():
    a = WordAnchors("same words", "same words")
    confidence, segmentation = a.describe(0, 0)
    assert confidence == 1.0 and segmentation is None


# ---- end to end through score() ---------------------------------------------

CFG = dict(scorer_version="1.1", strict=True)


def _subs(report):
    return {w.ground_truth_word: w for w in report.word_scores
            if w.alignment_type == "SUBSTITUTION"}


def test_clitic_join_scored_as_segmentation():
    r = score("It'll be here soon", "It 'll be here soon", config=ScorerConfig(**CFG))
    assert r.segmentation_count == 1
    seg = [w for w in r.word_scores if w.error_type == "SEGMENTATION"]
    assert len(seg) == 1
    assert seg[0].delta is None  # segmentation events get no Δ


def test_fragment_scored_as_segmentation_not_semantic():
    r = score("the net film was long", "the seine-net film was long",
              config=ScorerConfig(**CFG))
    assert r.segmentation_count == 1
    assert r.semantic_error_count == 0 and r.obvious_error_count == 0


def test_real_substitution_untouched_with_confidence():
    r = score("The parable of the vineyard", "The parable of the vinegar",
              config=ScorerConfig(**CFG))
    assert r.segmentation_count == 0
    ws = _subs(r)["vinegar"]
    assert ws.error_type == "SEMANTIC"
    assert ws.anchor_confidence is not None and ws.anchor_confidence > 0.4
    assert ws.delta is not None


def test_v10_path_has_no_anchor_fields():
    r = score("It'll be here soon", "It 'll be here soon",
              config=ScorerConfig(scorer_version="1.0"))
    assert r.segmentation_count == 0
    for w in r.word_scores:
        assert w.anchor_confidence is None
        assert w.error_type != "SEGMENTATION"
