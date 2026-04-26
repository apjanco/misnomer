from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

# Strips leading/trailing non-word characters (punctuation, whitespace).
# Applied after grouping so that "Macleod," → "Macleod" and "," → "" (filtered).
_BOUNDARY_RE = re.compile(r"^\W+|\W+$")


@dataclass(frozen=True)
class AlignmentItem:
    predicted_word: str
    ground_truth_word: str
    alignment_type: str


def _tokenize_with_hf(text: str, tokenizer) -> list[str]:
    """Group HuggingFace subword tokens into words using character offsets.

    Detects word boundaries by checking for whitespace gaps between adjacent
    token spans — robust across GPT-2-style (Ġ), SentencePiece (▁), and
    BERT (##-suffix) tokenizers.  Falls back to Ġ/▁ prefix heuristic for
    slow tokenizers that do not support ``return_offsets_mapping``.

    After grouping, strips leading/trailing non-word characters from each
    group (e.g. "Macleod," → "Macleod") and discards punctuation-only
    groups (e.g. "," → "").
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping")

    if offsets is not None:
        words: list[str] = []
        w_start: int | None = None
        w_end: int | None = None
        for start, end in offsets:
            if start == end:  # zero-length token (e.g. padding placeholder)
                continue
            # Byte-level BPE (Qwen2.5 / GPT-2 style) encodes the preceding
            # space *inside* the next token, so " horse" has offset (3, 9) —
            # flush-adjacent to "The" at (0, 3) with no numeric gap.
            # Detect this by checking whether text[start] is whitespace.
            is_new_word = (
                w_end is None
                or start > w_end
                or (start == w_end and start < len(text) and text[start].isspace())
            )
            if is_new_word:
                if w_start is not None:
                    stripped = _BOUNDARY_RE.sub("", text[w_start:w_end])
                    if stripped:
                        words.append(stripped)
                w_start, w_end = start, end
            else:
                w_end = end  # continuation: extend current word span
        if w_start is not None:
            stripped = _BOUNDARY_RE.sub("", text[w_start:w_end])
            if stripped:
                words.append(stripped)
        return words

    # Slow tokenizer fallback: detect word boundaries via Ġ / ▁ prefix.
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    words = []
    current: list[str] = []
    for tok in tokens:
        if tok.startswith(("\u0120", "\u2581")) and current:  # Ġ or ▁ → new word
            stripped = _BOUNDARY_RE.sub("", "".join(current))
            if stripped:
                words.append(stripped)
            current = []
        current.append(tok.lstrip("\u0120\u2581"))
    if current:
        stripped = _BOUNDARY_RE.sub("", "".join(current))
        if stripped:
            words.append(stripped)
    return words


def tokenize_words(text: str, tokenizer=None) -> list[str]:
    """Split *text* into a list of content words, discarding punctuation tokens.

    When a HuggingFace *tokenizer* is supplied the split follows the model's
    own byte-pair / sentencepiece boundaries, ensuring word spans are
    consistent with those used for perplexity computation.  Without a
    tokenizer a whitespace split is used and boundary punctuation is stripped
    from each token.
    """
    if tokenizer is not None:
        return _tokenize_with_hf(text, tokenizer)
    # Whitespace split + strip leading/trailing non-word characters.
    tokens = []
    for tok in text.strip().split():
        stripped = _BOUNDARY_RE.sub("", tok)
        if stripped:
            tokens.append(stripped)
    return tokens


def align_words(
    predicted_text: str,
    ground_truth_text: str,
    tokenizer=None,
) -> list[AlignmentItem]:
    pred = tokenize_words(predicted_text, tokenizer)
    gt = tokenize_words(ground_truth_text, tokenizer)

    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    aligned: list[AlignmentItem] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned.append(
                    AlignmentItem(
                        predicted_word=pred[i - 1],
                        ground_truth_word=gt[j - 1],
                        alignment_type="MATCH" if cost == 0 else "SUBSTITUTION",
                    )
                )
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append(
                AlignmentItem(
                    predicted_word=pred[i - 1],
                    ground_truth_word="",
                    alignment_type="INSERTION",
                )
            )
            i -= 1
            continue
        aligned.append(
            AlignmentItem(
                predicted_word="",
                ground_truth_word=gt[j - 1],
                alignment_type="DELETION",
            )
        )
        j -= 1

    aligned.reverse()
    return aligned


def char_edit_distance(left: str, right: str) -> int:
    return Levenshtein.distance(left, right)
