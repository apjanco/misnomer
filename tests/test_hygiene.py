"""Hygiene gate: no evaluation-corpus literal may appear in src/.

The scoring mechanisms are general; the corpus examples (the Vinegar Bible
page, IAM-derived words) exist only in tests as verification cases. A match
here means behavior may have been specialized to the examples it is supposed
to classify from first principles.

Spec-canonical illustrative pairs (horse/steed/house, insure/injure) are
allowed: they are documentation, not corpus content.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "misnomer"

# Word-bounded, case-insensitive. Plain substring matching false-positives
# ("parable" inside "comparable"), hence \b anchors.
FORBIDDEN = [
    "vinegar",
    "vineyard",
    "gaue",
    "priefts",
    "prieſts",
    "perswaded",
    "perſwaded",
    "faying",
    "doeft",
    "Baskett",
    "parable",
    "Luke",
]

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FORBIDDEN) + r")\b",
    re.IGNORECASE,
)


def test_no_corpus_literals_in_src():
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = _PATTERN.search(line)
            if match:
                offenders.append(f"{path.relative_to(SRC.parent.parent)}:{lineno}: {match.group(0)!r}")
    assert not offenders, "corpus literals leaked into src/:\n" + "\n".join(offenders)


def test_gate_pattern_detects_and_bounds():
    # The gate works…
    assert _PATTERN.search("the parable of the page")
    assert _PATTERN.search("Gaue is archaic")
    # …and word boundaries prevent the known false positive:
    assert not _PATTERN.search("scores are comparable across documents")
