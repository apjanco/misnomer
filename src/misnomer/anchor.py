"""Character-anchored word links: the alignment layer's confidence and
segmentation detector (v1.1).

The token-level DP alignment (aligner.py) stays untouched — WER/CER remain
computed on the field's standard basis. This module answers a question that
alignment cannot: for a word pair the DP called a SUBSTITUTION, do the
*characters* say it is really one word read as another — or a segmentation
event (two words joined, one word split, a fragment of a longer word)?

Method: align the two raw strings character-by-character (rapidfuzz
Levenshtein opcodes — deterministic, no models), then project whitespace
word spans onto the character alignment. Every word pair gets an anchor
confidence (fraction of identical aligned characters); pairs whose
character mass links across word boundaries are segmentation events, not
readings. This is OCR's own evaluation tradition (character-level anchoring
à la the ISRI analytic tools) applied to the substitution layer only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

_BOUNDARY_RE = re.compile(r"^\W+|\W+$")

# A cross-boundary link needs at least this many identical characters to
# count as evidence of joining/splitting (guards against coincidental
# single-letter anchors).
MIN_LINK_CHARS = 2
# A fragment: nearly all of the predicted word's characters anchor
# identically inside the ground-truth word (>= FRAGMENT_IDENT of pred), yet
# they cover at most FRAGMENT_COVER of it — a dropped morpheme, not a reading.
FRAGMENT_IDENT = 0.8
FRAGMENT_COVER = 0.6


@dataclass(frozen=True)
class _Span:
    text: str
    start: int  # char offset of the stripped token in the raw string
    end: int


def _token_spans(text: str) -> list[_Span]:
    """Whitespace tokens with char offsets, mirroring
    tokenize_words(text, tokenizer=None): boundary punctuation stripped,
    punctuation-only tokens dropped."""
    spans: list[_Span] = []
    pos = 0
    for raw in text.split():
        start = text.index(raw, pos)
        pos = start + len(raw)
        stripped = _BOUNDARY_RE.sub("", raw)
        if not stripped:
            continue
        inner = raw.index(stripped)
        spans.append(_Span(stripped, start + inner, start + inner + len(stripped)))
    return spans


class WordAnchors:
    """Character-anchored links between the words of two strings."""

    def __init__(self, predicted_text: str, ground_truth_text: str):
        self.pred_spans = _token_spans(predicted_text)
        self.gt_spans = _token_spans(ground_truth_text)

        p_tok = [None] * len(predicted_text)
        for idx, s in enumerate(self.pred_spans):
            for c in range(s.start, s.end):
                p_tok[c] = idx
        g_tok = [None] * len(ground_truth_text)
        for idx, s in enumerate(self.gt_spans):
            for c in range(s.start, s.end):
                g_tok[c] = idx

        # links[(p_idx, g_idx)] = [aligned_chars, identical_chars]
        self.links: dict[tuple[int, int], list[int]] = {}
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(predicted_text, ground_truth_text):
            if tag not in ("equal", "replace"):
                continue
            for k in range(i2 - i1):
                pi, gj = i1 + k, j1 + k
                p, g = p_tok[pi], g_tok[gj]
                if p is None or g is None:
                    continue
                cell = self.links.setdefault((p, g), [0, 0])
                cell[0] += 1
                if tag == "equal":
                    cell[1] += 1

    def pred_texts(self) -> list[str]:
        return [s.text for s in self.pred_spans]

    def gt_texts(self) -> list[str]:
        return [s.text for s in self.gt_spans]

    def _strong_partners(self, idx: int, side: int) -> list[int]:
        out = []
        for (p, g), (_, ident) in self.links.items():
            key = (p, g)[side]
            partner = (p, g)[1 - side]
            if key == idx and ident >= MIN_LINK_CHARS:
                out.append(partner)
        return sorted(set(out))

    def describe(self, p_idx: int, g_idx: int) -> tuple[float, str | None]:
        """(anchor_confidence, segmentation) for a substitution pair.

        confidence = identical anchored chars / max(word lengths).
        segmentation: "join" — the predicted word's characters anchor into
        multiple ground-truth words; "split" — the ground-truth word's
        characters anchor into multiple predicted words; "fragment" — the
        predicted word sits identically inside the ground-truth word but
        covers only part of it; None — an ordinary word-for-word reading.
        """
        p_len = len(self.pred_spans[p_idx].text)
        g_len = len(self.gt_spans[g_idx].text)
        aligned, ident = self.links.get((p_idx, g_idx), (0, 0))
        confidence = ident / max(p_len, g_len, 1)

        if len(self._strong_partners(p_idx, side=0)) > 1:
            return confidence, "join"
        if len(self._strong_partners(g_idx, side=1)) > 1:
            return confidence, "split"
        if (
            p_len < g_len
            and ident >= FRAGMENT_IDENT * p_len
            and ident <= FRAGMENT_COVER * g_len
        ):
            return confidence, "fragment"
        return confidence, None
