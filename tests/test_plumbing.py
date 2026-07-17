"""P4 end-to-end: gates, Δ, and context flags flow through score() under
scorer_version 1.1, and stay inert under 1.0."""

from __future__ import annotations

import pytest

from misnomer import StrictModeError, score
from misnomer.config import ScorerConfig


CFG_11 = dict(scorer_version="1.1", strict=True)

MIXED_PRED = "The prieſts gaue the steed 1771 coins"
MIXED_GT = "The priests gave the horse 1717 coins"


@pytest.fixture(scope="module")
def mixed_report():
    return score(MIXED_PRED, MIXED_GT, config=ScorerConfig(**CFG_11))


def test_mixed_document_class_counts(mixed_report):
    r = mixed_report
    assert r.archaizing_count == 2  # prieſts, gaue
    assert r.semantic_error_count == 1  # steed
    assert r.numeric_error_count == 1  # 1771
    assert r.garble_count == 0 and r.homoglyph_count == 0 and r.normalization_count == 0
    assert r.scorer_mode == "full" and r.strict is True


def test_mixed_document_deltas_and_context(mixed_report):
    subs = {w.ground_truth_word: w for w in mixed_report.word_scores
            if w.alignment_type == "SUBSTITUTION"}
    assert set(subs) == {"priests", "gave", "horse", "1717"}
    # positions in GT: The(0) priests(1) gave(2) the(3) horse(4) 1717(5) coins(6)
    assert subs["priests"].context_words == 1
    assert subs["gave"].context_words == 2
    assert subs["horse"].context_words == 4
    assert subs["1717"].context_words == 5
    for ws in subs.values():
        assert ws.delta is not None, f"missing Δ for {ws.ground_truth_word}"
    # the pull is real where it should be: steed-for-horse is a substitution
    # AWAY from the easy reading (negative Δ under a modern auditor)
    assert subs["horse"].delta < 0


def test_homoglyph_and_garble_document():
    r = score("He drew a mаp and wrote Tbe word",
              "He drew a map and wrote The word",
              config=ScorerConfig(**CFG_11))
    assert r.homoglyph_count == 1  # mаp with CYRILLIC а
    assert r.garble_count == 1  # Tbe
    types = {w.error_type for w in r.word_scores if w.alignment_type == "SUBSTITUTION"}
    assert types == {"HOMOGLYPH", "GARBLE"}


def test_normalization_direction_end_to_end():
    r = score("The priests read", "The prieſts read", config=ScorerConfig(**CFG_11))
    assert r.normalization_count == 1
    assert r.archaizing_count == 0


def test_v10_stays_inert():
    r = score(MIXED_PRED, MIXED_GT, config=ScorerConfig(scorer_version="1.0"))
    assert r.numeric_error_count == 0
    assert r.archaizing_count == 0
    assert r.normalization_count == 0
    assert r.garble_count == 0
    assert r.homoglyph_count == 0
    for w in r.word_scores:
        assert w.delta is None
        assert w.context_words is None
        assert w.error_type in {"SEMANTIC", "OBVIOUS", None}


def test_strict_gates_require_dictionary():
    cfg = ScorerConfig(scorer_version="1.1", strict=True, dictionary_lang="zz-nonexistent")
    with pytest.raises(StrictModeError, match="dictionary"):
        score("a word", "a ward", config=cfg)


def test_non_strict_broken_dictionary_degrades():
    cfg = ScorerConfig(scorer_version="1.1", strict=False, dictionary_lang="zz-nonexistent")
    r = score("The priefts read", "The priests read", config=cfg)
    # no dictionary -> no GARBLE on no evidence; falls through to similarity
    assert r.garble_count == 0
    sub = [w for w in r.word_scores if w.alignment_type == "SUBSTITUTION"][0]
    assert sub.error_type in {"SEMANTIC", "OBVIOUS"}


def test_v10_strict_does_not_require_dictionary():
    # gates inactive under 1.0: strict checks LM + embedder only
    r = score("a word", "a ward", config=ScorerConfig(scorer_version="1.0", strict=True))
    assert r.strict is True
