from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class AlignmentItem:
    predicted_word: str
    ground_truth_word: str
    alignment_type: str


def tokenize_words(text: str) -> list[str]:
    return [tok for tok in text.strip().split() if tok]


def align_words(predicted_text: str, ground_truth_text: str) -> list[AlignmentItem]:
    pred = tokenize_words(predicted_text)
    gt = tokenize_words(ground_truth_text)

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
