from __future__ import annotations

import logging
import unicodedata

from misnomer.config import ScorerConfig

log = logging.getLogger(__name__)


def classify_substitution(embedding_similarity: float, threshold: float = 0.5) -> str:
    """v1.0 classifier: similarity threshold only. Kept verbatim for
    scorer_version < 1.1 compatibility."""
    if embedding_similarity >= threshold:
        return "SEMANTIC"
    return "OBVIOUS"


# ---------------------------------------------------------------------------
# v1.1 gate tables — knowledge as data, overridable via ScorerConfig.
# Nothing here is specific to any evaluation corpus.
# ---------------------------------------------------------------------------

# Archaic character/sequence -> modern equivalent (Latin-script European
# print). These classes are the ones documented by transcription standards
# (e.g. the CATMuS guidelines); override via ScorerConfig.allograph_map.
DEFAULT_ALLOGRAPH_MAP: dict[str, str] = {
    "ſ": "s",
    "ß": "ss",
    "ẞ": "SS",
    "æ": "ae",
    "Æ": "AE",
    "œ": "oe",
    "Œ": "OE",
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

# Positional-usage equivalences: both characters are modern; the archaic
# *usage* (u-for-v, i-for-j, w-for-u as in "perswaded") is what varies.
# Folded for equality testing; direction is decided by dictionary membership.
# Note: Unicode casefolding already folds ſ->s and ß->ss underneath this
# layer, so those DEFAULT_ALLOGRAPH_MAP entries serve direction detection.
EQUIVALENCE_FOLD: dict[str, str] = {"v": "u", "j": "i", "w": "u"}

# Cross-script lookalikes, curated subset of Unicode TR#39 confusables.
# A word equal to its counterpart after this fold, but differing in script
# inventory, is a homoglyph substitution: invisible to the eye, corrosive
# to search, impossible for closed-inventory recognizers.
CONFUSABLES: dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "к": "k",
    "і": "i", "ј": "j", "ѕ": "s", "у": "y", "ԁ": "d", "ɡ": "g", "ν": "v",
    "ο": "o", "α": "a", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Υ": "Y", "Χ": "X",
}


def _has_digit(word: str) -> bool:
    return any(ch.isdigit() for ch in word)


def _scripts(word: str) -> frozenset[str]:
    scripts: set[str] = set()
    for ch in word:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            scripts.add(name.split(" ")[0] if name else "UNKNOWN")
    return frozenset(scripts)


def _confusable_fold(word: str) -> str:
    return "".join(CONFUSABLES.get(ch, ch) for ch in word)


def _allograph_fold(word: str, allograph_map: dict[str, str]) -> str:
    out = word.casefold()
    for archaic, modern in allograph_map.items():
        out = out.replace(archaic.casefold(), modern.casefold())
    for a, b in EQUIVALENCE_FOLD.items():
        out = out.replace(a, b)
    return out


def _has_archaic_chars(word: str, allograph_map: dict[str, str]) -> bool:
    return any(ch in word for ch in allograph_map)


class SpellDictionary:
    """Thin dictionary gate backend (pyspellchecker: pure Python, bundled
    wordlists incl. proper nouns). Unavailable -> gate reports itself off;
    the scorer decides whether that is an error (strict + gates) or a
    logged degradation."""

    def __init__(self, lang: str = "en"):
        self._sc = None
        code = lang.split(",")[0].split("_")[0].strip() or "en"
        try:
            from spellchecker import SpellChecker  # type: ignore

            self._sc = SpellChecker(language=code)
        except Exception as exc:  # ImportError or unknown language pack
            log.warning("SpellDictionary unavailable (%s: %s); GARBLE gate off.",
                        type(exc).__name__, exc)

    @property
    def available(self) -> bool:
        return self._sc is not None

    def known(self, word: str) -> bool:
        if self._sc is None:
            return True  # gate off: never classify GARBLE without evidence
        w = word.strip().casefold()
        if not w:
            return True
        return bool(self._sc.known([w]))


def classify_substitution_v2(
    predicted: str,
    ground_truth: str,
    embedding_similarity: float | None,
    cfg: ScorerConfig,
    dictionary: SpellDictionary | None = None,
) -> str | None:
    """Ordered gates, then the similarity threshold.

    NUMERIC -> HOMOGLYPH -> allograph (NORMALIZATION | ARCHAIZING) ->
    GARBLE -> SEMANTIC/OBVIOUS. Deterministic gates run first because
    embedding similarity is unreliable exactly where they apply (a non-word
    like a glyph-confused string can embed close to its target and would
    otherwise be counted as a semantic error).
    """
    allograph_map = cfg.allograph_map or DEFAULT_ALLOGRAPH_MAP

    # 1. Digit-bearing substitutions: embeddings and priors are both nearly
    # uninformative for numbers; classify by rule.
    if cfg.numeric_gate and (_has_digit(predicted) or _has_digit(ground_truth)):
        return "NUMERIC"

    # 2. Homoglyphs: identical after cross-script folding, different scripts.
    if predicted != ground_truth and _confusable_fold(predicted).casefold() == _confusable_fold(
        ground_truth
    ).casefold():
        if _scripts(predicted) != _scripts(ground_truth):
            return "HOMOGLYPH"

    # 3. Allograph pairs: equal under the archaic/modern fold. Direction by
    # definitive archaic characters first, dictionary membership second.
    if _allograph_fold(predicted, allograph_map) == _allograph_fold(ground_truth, allograph_map):
        pred_archaic = _has_archaic_chars(predicted, allograph_map)
        truth_archaic = _has_archaic_chars(ground_truth, allograph_map)
        if pred_archaic and not truth_archaic:
            return "ARCHAIZING"
        if truth_archaic and not pred_archaic:
            return "NORMALIZATION"
        if dictionary is not None and dictionary.available:
            pred_known = dictionary.known(predicted)
            truth_known = dictionary.known(ground_truth)
            if truth_known and not pred_known:
                return "ARCHAIZING"
            if pred_known and not truth_known:
                return "NORMALIZATION"
        return "NORMALIZATION"

    # 4. Non-words: visible damage, whatever the embedding thinks.
    if dictionary is not None and dictionary.available and not dictionary.known(predicted):
        return "GARBLE"

    # 5. Real-word substitution: the v1.0 semantic/obvious threshold.
    if embedding_similarity is None:
        return None
    return "SEMANTIC" if embedding_similarity >= cfg.semantic_threshold else "OBVIOUS"
