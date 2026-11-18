"""Generate the v1.1 behavior snapshot used as a regression oracle.

Run on the commit tagged ``table1-submitted``, BEFORE any v1.2 change to the
scorer (the scorer sources must be byte-identical to that tag):

    uv run python tests/make_v11_snapshot.py

The snapshot pins the complete report output of scorer_version="1.1"
(strict, CPU) as it stands at that commit, on the v1.0 pairs plus pairs that
exercise the v1.1 layers: every gate class, character anchoring,
segmentation and Δ, under the default "modern" tradition and, for a few
pairs, under "early-modern-latin-print". tests/test_v11_unchanged.py replays
the pairs and asserts equality, so that v1.2 work can never alter what
scorer_version="1.1" reports.

Scope of the guarantee. The submitted paper's scored files were produced
earlier in the life of "1.1" (commit 85aa792, on the mps device). Compared
with those files on a sample of 312 lines: error_type, context_words,
anchor_confidence, wer and cer are identical; delta agrees to within about
0.005 (mps against cpu arithmetic); perplexity, composite_score,
document_score and the surprisal and override fields differ and are pinned
here at their current cpu values.

The pairs are mostly invented; a few reproduce cases observed in the IAM
evaluation data (a split clitic, spaced initialisms, a line-edge fragment,
the "under censure" line).
"""

from __future__ import annotations

import json
from pathlib import Path

from misnomer import score
from misnomer.config import ScorerConfig

FIXTURES = Path(__file__).parent / "fixtures"
PAIRS_PATH = FIXTURES / "v11_pairs.json"
SNAPSHOT_PATH = FIXTURES / "v11_snapshot.json"

# (predicted, ground_truth) or (predicted, ground_truth, config overrides).
EXTRA_PAIRS: list[tuple] = [
    # real-word substitution of a visually unlike word, flanked by matches
    ("The board placed the plan under review.", "The board placed the plan under censure."),
    # two adjacent substitutions
    ("She disclosed that the policy holder answered.", "She declared that the policy under censure."),
    # reference with tokenised punctuation and split clitics
    ("It's a pity, he said, that you're late.", "It 's a pity , he said , that you 're late ."),
    ("It 's a pity , he said , that you 're late .", "It 's a pity , he said , that you 're late ."),
    # spaced abbreviation in the reference, joined in the prediction
    ("The MP spoke to the BBC today.", "The M P spoke to the B B C today."),
    # join and split of ordinary words
    ("He went in to the house today.", "He went into the house to day."),
    # fragment of a longer word
    ("An under was reached at last.", "An understanding was reached at last."),
    # line-edge fragments, both directions
    ("companionship was all he wanted", "panionship was all he wanted"),
    ("he wanted nothing but com", "he wanted nothing but companion-"),
    # numerals, incl. a year and a number with separators
    ("They paid 4,000 pounds in 1775.", "They paid 400,000 pounds in 1777."),
    # spelling variants that are both dictionary words
    ("The colors of the harbor were gray.", "The colours of the harbour were grey."),
    # non-word reference corrected to a word; word corrupted to a non-word
    ("We meet tomorrow at noon.", "We meet tommorow at noon."),
    ("We meet tomorow at noon.", "We meet tomorrow at noon."),
    # minim confusion between two dictionary words
    ("A modem design was chosen.", "A modern design was chosen."),
    # heavily damaged line with positional pairings
    ("Tbe hxvse ls ar al Mat te", "The house stood in the field that day"),
    # insertion-heavy output and a truncated output
    ("The house, the old house, stood in the wide field.", "The house stood in the field."),
    ("The house", "The house stood in the field."),
    # case change on a substituted word; apostrophes and hyphens inside words
    ("He said That the well-known writer's book didn't sell.", "He said that the well known writers book did not sell."),
    # allograph gate, both directions: the reference carries the archaic form / the prediction does
    ("The priests read the lesson aloud.", "The prie\u017fts read the le\u017f\u017fon aloud."),
    ("The prie\u017fts read the le\u017f\u017fon aloud.", "The priests read the lesson aloud."),
    # the same positional spellings under the default tradition and under a declared one
    ("upon the just man he looked", "vpon the iust man he looked"),
    ("upon the just man he looked", "vpon the iust man he looked", {"tradition": "early-modern-latin-print"}),
    ("vpon the iust man he looked", "upon the just man he looked", {"tradition": "early-modern-latin-print"}),
    ("The \u00e6ther was vvide and the horse stood", "The aether was wide and the house stood",
     {"tradition": "early-modern-latin-print"}),
]


def _round_floats(obj, ndigits: int = 6):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    return obj


def main() -> None:
    v10_pairs = [tuple(p) for p in json.loads((FIXTURES / "v10_pairs.json").read_text())]
    pairs = v10_pairs + EXTRA_PAIRS

    snapshots = []
    for predicted, ground_truth, *overrides in pairs:
        cfg = ScorerConfig(scorer_version="1.1", strict=True, **(overrides[0] if overrides else {}))
        report = score(predicted, ground_truth, config=cfg)
        snapshots.append(_round_floats(report.model_dump()))

    PAIRS_PATH.write_text(json.dumps(pairs, ensure_ascii=False, indent=1))
    SNAPSHOT_PATH.write_text(json.dumps(snapshots, ensure_ascii=False, indent=1))
    print(f"wrote {PAIRS_PATH.name} ({len(pairs)} pairs) and {SNAPSHOT_PATH.name}")


if __name__ == "__main__":
    main()
