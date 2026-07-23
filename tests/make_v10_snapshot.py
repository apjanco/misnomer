"""Generate the v1.0 behavior snapshot used as a regression oracle.

Run once on the delta-v1.1 branch BEFORE any behavioral change:

    uv run python tests/make_v10_snapshot.py

The snapshot pins the complete report output of scorer_version="1.0" on a
behaviorally diverse set of pairs. tests/test_v10_unchanged.py replays the
pairs and asserts equality (floats to 1e-4), guaranteeing that v1.1 changes
never alter v1.0 results.
"""

from __future__ import annotations

import json
from pathlib import Path

from misnomer import score
from misnomer.config import ScorerConfig

FIXTURES = Path(__file__).parent / "fixtures"
PAIRS_PATH = FIXTURES / "v10_pairs.json"
SNAPSHOT_PATH = FIXTURES / "v10_snapshot.json"

# Behaviorally diverse, deliberately generic pairs (predicted, ground_truth).
PAIRS: list[tuple[str, str]] = [
    # identical / trivial
    ("The house stood in the field.", "The house stood in the field."),
    ("", "The house stood in the field."),
    ("The house stood in the field.", ""),
    # canonical semantic vs obvious substitutions
    ("The steed stood in the field.", "The horse stood in the field."),
    ("The house stood in the field.", "The horse stood in the field."),
    # multiple substitutions in one line
    ("A large hound slept near the gate.", "A small hound slept near the door."),
    # insertion / deletion
    ("The old house stood in the field.", "The house stood in the field."),
    ("The house stood in field.", "The house stood in the field."),
    # non-word garble
    ("The hxvse stood in the field.", "The house stood in the field."),
    # digits
    ("Founded in 1771, the press closed.", "Founded in 1717, the press closed."),
    # unicode: long s, accents, homoglyph (Cyrillic а in 'mаp')
    ("The prieſts read the leſſon aloud.", "The priests read the lesson aloud."),
    ("He drew a mаp of the region.", "He drew a map of the region."),
    ("She sang a believèd refrain.", "She sang a beloved refrain."),
    # case-only and punctuation-spacing differences
    ("the house stood in the field.", "The house stood in the field."),
    ("The house stood in the field .", "The house stood in the field."),
    # refusal and boilerplate paths
    ("I'm sorry, I cannot transcribe this image.", "The house stood in the field."),
    ("```\nThe house stood in the field.\n```", "The house stood in the field."),
    ("Here is the transcription: The house stood in the field.", "The house stood in the field."),
    # rare word substituted by frequent word (pull-shaped)
    ("The order of the garter was renewed.", "The ordre of the garter was renewed."),
    # longer passage with one mid-document substitution
    (
        "In the second year the committee published its findings, and the town "
        "council voted to preserve the bridge for another century.",
        "In the second year the committee published its findings, and the town "
        "counsel voted to preserve the bridge for another century.",
    ),
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
    FIXTURES.mkdir(exist_ok=True)
    cfg = ScorerConfig(scorer_version="1.0", local_files_only=False, allow_download=True)

    snapshots = []
    for predicted, ground_truth in PAIRS:
        report = score(predicted, ground_truth, config=cfg)
        snapshots.append(_round_floats(report.model_dump()))

    PAIRS_PATH.write_text(json.dumps(PAIRS, ensure_ascii=False, indent=1))
    SNAPSHOT_PATH.write_text(json.dumps(snapshots, ensure_ascii=False, indent=1))
    print(f"wrote {PAIRS_PATH.name} ({len(PAIRS)} pairs) and {SNAPSHOT_PATH.name}")


if __name__ == "__main__":
    main()
