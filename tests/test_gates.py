"""v1.1 classification gates: NUMERIC -> HOMOGLYPH -> allograph -> GARBLE ->
SEMANTIC/OBVIOUS. Corpus examples appear here and only here; the gates
themselves are general rules over character classes, script inventories,
allograph tables, and a dictionary."""

from __future__ import annotations

import pytest

from misnomer.classifier import (
    SpellDictionary,
    classify_substitution,
    classify_substitution_v2,
)
from misnomer.config import ScorerConfig
from misnomer.models.embedder import Embedder


@pytest.fixture(scope="module")
def cfg() -> ScorerConfig:
    return ScorerConfig(scorer_version="1.1")


@pytest.fixture(scope="module")
def dictionary() -> SpellDictionary:
    d = SpellDictionary("en")
    assert d.available, "pyspellchecker required (uv sync --extra gates)"
    return d


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    e = Embedder(ScorerConfig())
    assert e.is_model_backed
    return e


def classify(pred, truth, cfg, dictionary, embedder):
    sim = embedder.similarity(pred, truth)
    return classify_substitution_v2(pred, truth, sim, cfg, dictionary)


# ---- the fixture matrix -----------------------------------------------------

MATRIX = [
    ("vineyard", "vinegar", "SEMANTIC"),      # the fluent correction
    ("steed", "horse", "SEMANTIC"),           # canonical semantic neighbor
    ("house", "horse", "OBVIOUS"),            # canonical obvious substitution
    ("gaue", "gave", "ARCHAIZING"),           # u-for-v period usage, direction by dictionary
    ("aſk", "ask", "ARCHAIZING"),             # long-s extended against the page
    ("priests", "prieſts", "NORMALIZATION"),  # long-s modernized (Kraken's systematic case)
    ("persuaded", "perſwaded", "NORMALIZATION"),
    ("priefts", "priests", "GARBLE"),         # f-for-long-s misreading: non-word
    ("Tbe", "The", "GARBLE"),                 # classic glyph garble
    ("faying", "saying", "GARBLE"),           # non-word despite sim > threshold
    ("1771", "1717", "NUMERIC"),              # digit swap: rule, not embedding
    ("vinegаr", "vinegar", "HOMOGLYPH"),      # Cyrillic а (U+0430)
    ("Luкe", "Luke", "HOMOGLYPH"),            # Cyrillic к (U+043A)
    ("counsel", "council", "SEMANTIC"),       # real-word near-homophone
]


@pytest.mark.parametrize("pred,truth,expected", MATRIX, ids=lambda v: str(v))
def test_fixture_matrix(pred, truth, expected, cfg, dictionary, embedder):
    got = classify(pred, truth, cfg, dictionary, embedder)
    assert got == expected, f"{pred!r} vs {truth!r}: expected {expected}, got {got}"


# ---- gate-specific behavior -------------------------------------------------

def test_garble_gate_rescues_high_similarity_nonwords(dictionary, embedder, cfg):
    """priefts embeds ABOVE the semantic threshold (0.378 measured) — without
    the dictionary gate it would be counted as a semantic error. This is the
    mechanism that inflated smooth-substitution counts for noisy systems."""
    sim = embedder.similarity("priefts", "priests")
    assert sim >= cfg.semantic_threshold  # the trap exists...
    got = classify_substitution_v2("priefts", "priests", sim, cfg, dictionary)
    assert got == "GARBLE"  # ...and the gate closes it


def test_allograph_direction_is_symmetric(cfg, dictionary):
    assert classify_substitution_v2("priests", "prieſts", 0.9, cfg, dictionary) == "NORMALIZATION"
    assert classify_substitution_v2("prieſts", "priests", 0.9, cfg, dictionary) == "ARCHAIZING"


def test_numeric_gate_configurable(dictionary):
    cfg_off = ScorerConfig(scorer_version="1.1", numeric_gate=False)
    got = classify_substitution_v2("1771", "1717", 0.656, cfg_off, dictionary)
    assert got != "NUMERIC"


def test_dictionary_unavailable_degrades_without_garble():
    """No dictionary -> never classify GARBLE on no evidence; word falls
    through to the similarity threshold."""
    cfg = ScorerConfig(scorer_version="1.1")
    broken = SpellDictionary("zz-nonexistent")
    assert not broken.available
    got = classify_substitution_v2("priefts", "priests", 0.378, cfg, broken)
    assert got == "SEMANTIC"  # honest degradation, to be guarded by strict mode in P4


def test_custom_allograph_map_overrides_default(dictionary):
    cfg = ScorerConfig(scorer_version="1.1", allograph_map={"þ": "th"})
    assert classify_substitution_v2("the", "þe", 0.9, cfg, dictionary) == "NORMALIZATION"
    # default table no longer applies under the override (æ is map-only;
    # ſ would still fold via Unicode casefolding, a lower layer than the map):
    assert classify_substitution_v2("aether", "æther", 0.9, cfg, dictionary) != "NORMALIZATION"


def test_unicode_casefold_layer_is_below_the_map(dictionary):
    """ſ->s comes from str.casefold(), not the allograph table; the table
    entry exists for direction detection. Equality therefore holds even
    under a custom map, and the pair classifies NORMALIZATION via the
    direction-ambiguous fallthrough."""
    cfg = ScorerConfig(scorer_version="1.1", allograph_map={"þ": "th"})
    got = classify_substitution_v2("priests", "prieſts", 0.9, cfg, dictionary)
    assert got == "NORMALIZATION"


def test_v1_classifier_untouched():
    assert classify_substitution(0.5, 0.35) == "SEMANTIC"
    assert classify_substitution(0.2, 0.35) == "OBVIOUS"
