"""Character and word error rates on declared bases.

The classic rates answer "how many edits turn the transcription into the
reference?" — but the answer depends on what counts as the text. Comparing
raw strings charges a system for case, punctuation and spacing; comparing
normalised words does not. Reporting a character rate on one footing and a
word rate on another makes the two incomparable. This module therefore
computes BOTH rates from the SAME sequence, on two named bases, and returns
integer counts so that corpus-level figures are exact sums rather than
averages of ratios.

Definition ``error-rates/1``
----------------------------
Both strings are first brought to Unicode Normalization Form C (canonical
equivalence only; never a compatibility form, which would turn long s into
s). If a reference convention is given (see ``conventions.py``) it is then
applied to both strings.

``verbatim``
    Every run of whitespace (as ``str.split`` defines it, line breaks
    included) becomes one space and the ends are trimmed. Nothing else is
    altered: case, punctuation and every other character count.
    Characters = that string; words = its space-separated tokens.

``content``
    The string is split on whitespace. In each token, typographic
    apostrophes (U+2019, U+02BC) are read as the apostrophe U+0027. Then
    everything before the first and after the last letter or number of the
    token is removed (Unicode general categories L and N: letters, decimal
    digits, and also fractions, superscripts and enclosed numbers), except
    that combining marks directly following that last letter or number stay
    with it — so an accent or an abbreviation mark is never dropped, while
    punctuation and symbols at the edges always are. A token with no letter
    or number is discarded: punctuation and symbols standing alone are not
    words. The token is lower-cased with ``str.lower`` (not casefold, which
    would also fold long s and ß) and brought back to Form C. Characters
    inside a token — hyphens, dashes, apostrophes, full stops — are kept:
    "well-known" is one word and differs from "well known".
    Characters = the tokens joined by single spaces; words = the tokens.

For each basis: ``char_edits`` and ``word_edits`` are Levenshtein distances
(insertions, deletions and substitutions at unit cost; no transpositions)
between the prediction and the reference at character and at token level;
``ref_chars`` and ``ref_words`` are the reference lengths. A rate is edits
divided by reference length; over a corpus, the sum of edits divided by the
sum of reference lengths.

Using one sequence for both rates removes the artefact of mixed bases (a
word rate below the character rate because only the characters were charged
for layout). It does not make WER >= CER a law: a long insertion inside one
word costs many characters and one word.

Whether the prediction passed to :func:`error_rates` has been cleaned
beforehand is the caller's decision and should be stated with the figures:
the functions here alter nothing beyond what is described above.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz.distance import Levenshtein

from misnomer.conventions import ReferenceConvention, resolve_convention

DEFINITION = "error-rates/1"
UNICODE_FORM = "NFC"
BASES = ("verbatim", "content")

# Typographic apostrophes read as U+0027 on the content basis.
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'"})


class ErrorCounts(BaseModel):
    """Edit counts and reference lengths for one basis."""

    model_config = ConfigDict(frozen=True)

    char_edits: int = Field(ge=0)
    ref_chars: int = Field(ge=0)
    word_edits: int = Field(ge=0)
    ref_words: int = Field(ge=0)

    @property
    def cer(self) -> float | None:
        """Character error rate, or None when the reference has no characters."""
        return self.char_edits / self.ref_chars if self.ref_chars else None

    @property
    def wer(self) -> float | None:
        """Word error rate, or None when the reference has no words."""
        return self.word_edits / self.ref_words if self.ref_words else None


class ConventionIdentity(BaseModel):
    """Which reference convention a figure was computed under."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    rules_sha256: str


class ErrorRates(BaseModel):
    """Counts on both bases, with everything needed to say how they were made."""

    model_config = ConfigDict(frozen=True)

    definition: str = DEFINITION
    unicode_form: str = UNICODE_FORM
    # None when the strings were compared as supplied.
    convention: ConventionIdentity | None = None
    verbatim: ErrorCounts
    content: ErrorCounts


def _is_letter_or_number(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LN"


def _strip_edges(token: str) -> str:
    """Keep *token* from its first to its last letter or number, plus the
    combining marks that directly follow that last one.

    'café.' becomes 'café' in any normal form, 'zuͤ' stays 'zuͤ', and a mark
    sitting on trailing punctuation ('taxi.̶') goes with the punctuation.
    Returns '' when the token holds no letter or number.
    """
    start = 0
    while start < len(token) and not _is_letter_or_number(token[start]):
        start += 1
    end = len(token)
    while end > start and not _is_letter_or_number(token[end - 1]):
        end -= 1
    while end < len(token) and unicodedata.category(token[end])[0] == "M":
        end += 1
    return token[start:end]


def verbatim_view(text: str) -> tuple[str, list[str]]:
    """(characters, words) of *text* on the verbatim basis."""
    chars = " ".join(text.split())
    return chars, (chars.split(" ") if chars else [])


def content_view(text: str) -> tuple[str, list[str]]:
    """(characters, words) of *text* on the content basis."""
    words = []
    for raw in text.split():
        token = _strip_edges(raw.translate(_APOSTROPHES))
        if token:
            words.append(unicodedata.normalize(UNICODE_FORM, token.lower()))
    return " ".join(words), words


def _count(pred_view: tuple[str, list[str]], ref_view: tuple[str, list[str]]) -> ErrorCounts:
    pred_chars, pred_words = pred_view
    ref_chars, ref_words = ref_view
    return ErrorCounts(
        char_edits=Levenshtein.distance(pred_chars, ref_chars),
        ref_chars=len(ref_chars),
        word_edits=Levenshtein.distance(pred_words, ref_words),
        ref_words=len(ref_words),
    )


def _text(value: str | None, label: str) -> str:
    """None is an absent answer (the empty string); anything else must be text."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or None, got {type(value).__name__}")
    return value


def error_rates(
    predicted: str | None,
    ground_truth: str | None,
    convention: "str | dict | ReferenceConvention | None" = None,
) -> ErrorRates:
    """Edit counts for one prediction/reference pair on both bases.

    *convention* is a built-in name, an inline dict, a
    :class:`~misnomer.conventions.ReferenceConvention`, or None (compare the
    strings as supplied). It is applied to both strings.
    """
    resolved = resolve_convention(convention)
    pred = unicodedata.normalize(UNICODE_FORM, _text(predicted, "predicted"))
    ref = unicodedata.normalize(UNICODE_FORM, _text(ground_truth, "ground_truth"))
    if resolved is not None:
        pred = resolved.apply(pred)
        ref = resolved.apply(ref)
    return ErrorRates(
        convention=ConventionIdentity(**resolved.identity) if resolved is not None else None,
        verbatim=_count(verbatim_view(pred), verbatim_view(ref)),
        content=_count(content_view(pred), content_view(ref)),
    )


def pool(items: Iterable[ErrorRates]) -> ErrorRates:
    """Sum the counts of several pairs (corpus-level, micro-averaged).

    All items must share one definition, Unicode form and convention;
    mixing them is an error, because the sum would describe nothing.
    """
    items = list(items)
    if not items:
        raise ValueError("pool() needs at least one ErrorRates")
    for item in items:
        if not isinstance(item, ErrorRates):
            raise TypeError(f"pool() takes ErrorRates, got {type(item).__name__}")
    first = items[0]
    for item in items[1:]:
        if (item.definition, item.unicode_form, item.convention) != (
            first.definition, first.unicode_form, first.convention
        ):
            raise ValueError("cannot pool error rates computed under different definitions or conventions")

    def total(basis: str) -> ErrorCounts:
        counts = [getattr(item, basis) for item in items]
        return ErrorCounts(
            char_edits=sum(c.char_edits for c in counts),
            ref_chars=sum(c.ref_chars for c in counts),
            word_edits=sum(c.word_edits for c in counts),
            ref_words=sum(c.ref_words for c in counts),
        )

    return ErrorRates(
        definition=first.definition,
        unicode_form=first.unicode_form,
        convention=first.convention,
        verbatim=total("verbatim"),
        content=total("content"),
    )
