from __future__ import annotations

from html import escape

from .report import SemanticErrorReport


def highlight(
    report: SemanticErrorReport,
    semantic_color: str = "#FF6B6B",
    obvious_color: str = "#FFD93D",
    threshold: float = 0.3,
) -> str:
    fragments: list[str] = []
    for ws in report.word_scores:
        token = ws.predicted_word or ws.ground_truth_word
        token = escape(token)
        if ws.error_type == "SEMANTIC" and ws.composite_score >= threshold:
            fragments.append(f'<span style="background:{semantic_color}">{token}</span>')
        elif ws.error_type == "OBVIOUS" and ws.composite_score >= threshold:
            fragments.append(f'<span style="background:{obvious_color}">{token}</span>')
        else:
            fragments.append(token)
    body = " ".join(fragments)
    return f"<html><body><p>{body}</p></body></html>"
