"""v1.1 classification gates: NUMERIC -> allograph -> HOMOGLYPH -> GARBLE ->
SEMANTIC/OBVIOUS. Corpus examples appear here and only here; the gates are
general rules over Unicode identity (compatibility caseless matching),
TR39 visual skeletons, a declared orthographic-tradition profile, and a
dictionary. No convention is applied silently: period equivalences exist
only under an explicitly selected tradition."""

from __future__ import annotations

import pytest

from misnomer.classifier import (
    SpellDictionary,
    classify_substitution,
    classify_substitution_v2,
)
from misnomer.config import ScorerConfig
from misnomer.models.embedder import Embedder
from misnomer.traditions import builtin_traditions, resolve_tradition

MODERN = ScorerConfig(scorer_version="1.1")  # default tradition: modern
EARLY = ScorerConfig(scorer_version="1.1", tradition="early-modern-latin-print")


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
# (predicted, ground truth, config, expected)

MATRIX = [
    # later gates, unchanged by tradition
    ("vineyard", "vinegar", MODERN, "SEMANTIC"),   # the fluent correction
    ("steed", "horse", MODERN, "SEMANTIC"),        # canonical semantic neighbor
    ("house", "horse", MODERN, "OBVIOUS"),         # canonical obvious substitution
    ("counsel", "council", MODERN, "SEMANTIC"),    # real-word near-homophone
    ("Tbe", "The", MODERN, "GARBLE"),              # classic glyph garble
    ("faying", "saying", MODERN, "GARBLE"),        # non-word despite sim > threshold
    ("1771", "1717", MODERN, "NUMERIC"),           # digit swap: rule, not embedding
    ("Ⅳ", "IV", MODERN, "NUMERIC"),                # numeral display form: isnumeric
    # Unicode identity layer (profile-independent: compatibility characters)
    ("priests", "prieſts", MODERN, "NORMALIZATION"),  # long s modernized (Kraken's case)
    ("prieſts", "priests", MODERN, "ARCHAIZING"),     # long s extended against the page
    ("aſk", "ask", MODERN, "ARCHAIZING"),
    # tradition layer: period equivalences only under a declared profile
    ("gaue", "gave", EARLY, "ARCHAIZING"),         # u-for-v period usage
    ("gaue", "gave", MODERN, "GARBLE"),            # same pair on a modern document
    ("persuaded", "perſwaded", EARLY, "NORMALIZATION"),
    ("will", "vvill", EARLY, "NORMALIZATION"),     # vv-for-w
    ("vs", "us", EARLY, "NORMALIZATION"),          # direction undecidable: weaker claim
    ("vacum", "vacuum", EARLY, "ARCHAIZING"),      # documented folding cost (uu -> u)
    ("vacum", "vacuum", MODERN, "GARBLE"),         # the modern profile is immune
    # TR39 visual layer, on the raw glyphs
    ("vinegаr", "vinegar", MODERN, "HOMOGLYPH"),   # Cyrillic а (U+0430), identical by design
    ("bam", "barn", MODERN, "HOMOGLYPH"),          # the minim class: TR39 maps m ~ rn
    ("priefts", "prieſts", MODERN, "HOMOGLYPH"),   # long-s/f likeness per TR39
    ("Luкe", "Luke", MODERN, "GARBLE"),            # TR39 grades: к is NOT a k lookalike
    ("priefts", "priests", MODERN, "GARBLE"),      # f-for-long-s against modernized truth
]


@pytest.mark.parametrize(
    "pred,truth,cfg,expected",
    MATRIX,
    ids=[f"{p}~{t}[{resolve_tradition(c).name[:6]}]" for p, t, c, _ in MATRIX],
)
def test_fixture_matrix(pred, truth, cfg, expected, dictionary, embedder):
    got = classify(pred, truth, cfg, dictionary, embedder)
    assert got == expected, f"{pred!r} vs {truth!r}: expected {expected}, got {got}"


# ---- gate-specific behavior -------------------------------------------------

def test_garble_gate_rescues_high_similarity_nonwords(dictionary, embedder):
    """priefts embeds ABOVE the semantic threshold (0.378 measured) — without
    the dictionary gate it would be counted as a semantic error. This is the
    mechanism that inflated smooth-substitution counts for noisy systems."""
    sim = embedder.similarity("priefts", "priests")
    assert sim >= MODERN.semantic_threshold  # the trap exists...
    got = classify_substitution_v2("priefts", "priests", sim, MODERN, dictionary)
    assert got == "GARBLE"  # ...and the gate closes it


def test_identity_precedes_appearance(dictionary):
    """prieſts/priests are the same text (compatibility equivalence) AND
    visually confusable in the ſ/f direction (TR39). Identity must win:
    the pair classifies as an allograph change, never HOMOGLYPH."""
    assert classify_substitution_v2("priests", "prieſts", 0.9, MODERN, dictionary) == "NORMALIZATION"


def test_numeric_gate_configurable(dictionary):
    cfg_off = ScorerConfig(scorer_version="1.1", numeric_gate=False)
    got = classify_substitution_v2("1771", "1717", 0.656, cfg_off, dictionary)
    assert got != "NUMERIC"


def test_dictionary_unavailable_degrades_without_garble():
    """No dictionary -> never classify GARBLE on no evidence; word falls
    through to the similarity threshold."""
    broken = SpellDictionary("zz-nonexistent")
    assert not broken.available
    got = classify_substitution_v2("priefts", "priests", 0.378, MODERN, broken)
    assert got == "SEMANTIC"  # honest degradation, guarded by strict mode


def test_legacy_allograph_map_becomes_custom_profile(dictionary):
    cfg = ScorerConfig(scorer_version="1.1", allograph_map={"þ": "th"})
    profile = resolve_tradition(cfg)
    assert profile.name == "custom-allograph-map"
    assert classify_substitution_v2("the", "þe", 0.9, cfg, dictionary) == "NORMALIZATION"
    # the custom profile does not include ae-digraph equivalence; the pair
    # then falls to the TR39 visual layer (æ ~ ae by the standard's data):
    assert classify_substitution_v2("aether", "æther", 0.9, cfg, dictionary) == "HOMOGLYPH"


def test_compat_layer_is_below_any_profile(dictionary):
    """ſ->s comes from Unicode compatibility folding, a layer beneath every
    tradition profile; equality holds even under a custom map, direction by
    the compatibility marker on the ground-truth side."""
    cfg = ScorerConfig(scorer_version="1.1", allograph_map={"þ": "th"})
    got = classify_substitution_v2("priests", "prieſts", 0.9, cfg, dictionary)
    assert got == "NORMALIZATION"


def test_inline_tradition_dict(dictionary):
    cfg = ScorerConfig(
        scorer_version="1.1",
        tradition={"name": "test-inline", "equivalences": [["þ", "th"]], "markers": ["þ"]},
    )
    assert resolve_tradition(cfg).name == "test-inline"
    assert classify_substitution_v2("the", "þe", 0.9, cfg, dictionary) == "NORMALIZATION"


def test_unknown_tradition_raises():
    with pytest.raises(ValueError, match="unknown tradition"):
        resolve_tradition(ScorerConfig(scorer_version="1.1", tradition="no-such-era"))


def test_builtin_traditions_ship():
    names = set(builtin_traditions())
    assert {"modern", "early-modern-latin-print"} <= names


def test_v1_classifier_untouched():
    assert classify_substitution(0.5, 0.35) == "SEMANTIC"
    assert classify_substitution(0.2, 0.35) == "OBVIOUS"
