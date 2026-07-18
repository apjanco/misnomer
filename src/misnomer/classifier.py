from __future__ import annotations

import logging

from misnomer.config import ScorerConfig
from misnomer.traditions import TraditionProfile, resolve_tradition
from misnomer.unicode_ops import skeleton

log = logging.getLogger(__name__)


def classify_substitution(embedding_similarity: float, threshold: float = 0.5) -> str:
    """v1.0 classifier: similarity threshold only. Kept verbatim for
    scorer_version < 1.1 compatibility."""
    if embedding_similarity >= threshold:
        return "SEMANTIC"
    return "OBVIOUS"


# ---------------------------------------------------------------------------
# v1.1 gates. Knowledge sources, by layer:
#   character identity  -> the Unicode Standard (compatibility caseless
#                          matching) and UTS #39 skeletons; see unicode_ops
#   orthographic usage  -> a declared TraditionProfile; see traditions
#   lexicon             -> SpellDictionary below
# Nothing here is specific to any evaluation corpus (enforced by
# tests/test_hygiene.py), and no convention is applied silently: period
# equivalences exist only under an explicitly selected tradition.
# ---------------------------------------------------------------------------


def _has_numeral(word: str) -> bool:
    # isnumeric (not isdigit) so numeral display forms — Roman-numeral
    # codepoints, vulgar fractions, circled digits — are caught by rule
    # rather than dragged into the allograph gate by compatibility folding.
    return any(ch.isnumeric() for ch in word)


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
    profile: TraditionProfile | None = None,
) -> str | None:
    """Ordered gates, then the similarity threshold.

    NUMERIC -> allograph (NORMALIZATION | ARCHAIZING) -> HOMOGLYPH ->
    GARBLE -> SEMANTIC/OBVIOUS.

    Character identity precedes visual similarity: compatibility
    equivalence says what a character IS (long s is s), the TR39 skeleton
    says what it LOOKS like (long s resembles f). A substitution that is
    the same text in a different display form must classify as an
    allograph change, never as a lookalike. Deterministic gates run before
    the embedding threshold because similarity is unreliable exactly where
    they apply (a glyph-confused non-word can embed close to its target
    and would otherwise be counted as a semantic error).
    """
    if profile is None:
        profile = resolve_tradition(cfg)

    # 1. Numeral-bearing substitutions: embeddings and priors are both
    # nearly uninformative for numbers; classify by rule.
    if cfg.numeric_gate and (_has_numeral(predicted) or _has_numeral(ground_truth)):
        return "NUMERIC"

    # 2. Allograph pairs: same text under Unicode compatibility caseless
    # matching plus the declared tradition's equivalences. Direction: the
    # side carrying compatibility characters (or the profile's markers) is
    # the archaic side; otherwise lexicon membership decides; if direction
    # remains undecidable (e.g. both words in the lexicon), report the
    # weaker claim rather than guess — counted and audited in validation.
    if profile.fold(predicted) == profile.fold(ground_truth):
        pred_marked = profile.marks_archaic(predicted)
        truth_marked = profile.marks_archaic(ground_truth)
        if pred_marked and not truth_marked:
            return "ARCHAIZING"
        if truth_marked and not pred_marked:
            return "NORMALIZATION"
        if dictionary is not None and dictionary.available:
            pred_known = dictionary.known(predicted)
            truth_known = dictionary.known(ground_truth)
            if truth_known and not pred_known:
                return "ARCHAIZING"
            if pred_known and not truth_known:
                return "NORMALIZATION"
        return "NORMALIZATION"

    # 3. Homoglyphs: visually confusable as written, by the standard's own
    # judgment (TR39 skeleton equality on the raw glyphs). Covers both
    # cross-script lookalikes and the same-script minim class (rn/m).
    if predicted != ground_truth and skeleton(predicted) == skeleton(ground_truth):
        return "HOMOGLYPH"

    # 4. Non-words: visible damage, whatever the embedding thinks.
    if dictionary is not None and dictionary.available and not dictionary.known(predicted):
        return "GARBLE"

    # 5. Real-word substitution: the v1.0 semantic/obvious threshold.
    if embedding_similarity is None:
        return None
    return "SEMANTIC" if embedding_similarity >= cfg.semantic_threshold else "OBVIOUS"
