from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

_REFUSAL_RE = re.compile(
    r"^\s*("
    r"i(?:'m| am) (?:sorry|unable|not able)"
    r"|i can(?:not|'t)"
    r"|i (?:apologize|cannot)"
    r"|as an? (?:ai|language model|llm|assistant)"
    r"|(?:i'm )?(?:unable|not able) to (?:transcribe|read|process|access|view|see)"
    r"|(?:sorry|apologies)[,.]? i"
    r")",
    re.IGNORECASE,
)


def is_refusal(text: str) -> bool:
    """Return True if *text* looks like a VLM refusal rather than a transcription."""
    return bool(_REFUSAL_RE.match(text.strip()))


# ---------------------------------------------------------------------------
# Boilerplate stripping
# ---------------------------------------------------------------------------

# Markdown fences (opening and closing), e.g. ```markdown ... ```
_FENCE_RE = re.compile(r"^```[a-z]*\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)

# Preamble phrases at the start of the text — longer/more-specific patterns first
_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"here (?:is|are)(?: the)? (?:transcri(?:bed|ption)|extracted|following)(?: text)?[:\s]*"
    r"|i found(?: the)? (?:transcri(?:bed|ption)|extracted|following)(?: text)?[:\s]*"
    r"|(?:the )?(?:transcri(?:bed|ption)|extracted|following)(?: text)?(?:\s+(?:is|reads|below))?[:\s]*"
    r"|here (?:is|are)(?: the)?[:\s]*"
    r"|i found(?: the)?[:\s]*"
    r"|below (?:is|are)(?: the)?(?: (?:transcri(?:bed|ption)|text))?[:\s]*"
    r")",
    re.IGNORECASE,
)

# Trailing conversational remarks
_TRAILING_RE = re.compile(
    r"\s*(?:"
    r"i hope this helps[.!]?"
    r"|please (?:let me know|note)[^.]*[.!]?"
    r"|(?:let me know if|if you (?:need|have))[^.]*[.!]?"
    r"|note[:\s]+[^\n]*$"
    r")\s*$",
    re.IGNORECASE,
)


@dataclass
class PreprocessResult:
    text: str
    is_refusal: bool = False
    transformations: list[str] = field(default_factory=list)


def preprocess(text: str | None) -> PreprocessResult:
    """Detect refusals and strip VLM boilerplate from *text*.

    Returns a :class:`PreprocessResult` with the cleaned text and a record
    of which transformations were applied.
    """
    if not text:
        return PreprocessResult(text=text or "")

    if is_refusal(text):
        return PreprocessResult(text=text, is_refusal=True, transformations=["refusal_detected"])

    cleaned = text
    transformations: list[str] = []

    # Strip markdown fences
    fence_match = _FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1)
        transformations.append("strip_markdown_fence")

    # Strip preamble phrases
    stripped = _PREAMBLE_RE.sub("", cleaned)
    if stripped != cleaned:
        cleaned = stripped
        transformations.append("strip_preamble")

    # Strip trailing conversational remarks
    stripped = _TRAILING_RE.sub("", cleaned)
    if stripped != cleaned:
        cleaned = stripped
        transformations.append("strip_trailing_remark")

    return PreprocessResult(text=cleaned.strip(), transformations=transformations)
