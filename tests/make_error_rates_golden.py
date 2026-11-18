"""Generate the golden fixture for error-rate definition "error-rates/1".

    uv run python tests/make_error_rates_golden.py

Each pair is scored without a convention and under the built-in
"tokenised-english" convention; the fixture stores the eight integers of
each result and tests/test_rates.py replays them. Together with the unit
tests and the pinned rule hash in tests/test_conventions.py this is what
holds the definition still: a change to the tokenisation, the Unicode
handling or the convention rules should fail a test and be released as a
new definition or convention version instead of silently moving published
figures. Regenerate ONLY together with such a version bump.

The pairs are mostly invented; a few reproduce layout cases observed in the
IAM evaluation data ("dragged on ,with", "return... ...which shows").
"""

from __future__ import annotations

import json
from pathlib import Path

from misnomer.rates import BASES, DEFINITION, error_rates

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "error_rates_golden.json"
CONVENTIONS = [None, "tokenised-english"]

# (predicted, reference).
PAIRS: list[tuple[str, str]] = [
    ("The house stood in the field.", "The house stood in the field."),
    ("", "The house stood in the field."),
    ("The house stood in the field.", ""),
    ("", ""),
    ("the house stood in the field", "The house stood in the field."),
    ("The horse stood in the field.", "The house stood in the field."),
    ("The old house stood in the field.", "The house stood in the field."),
    ("The house stood in field.", "The house stood in the field."),
    ("The  house\nstood\tin the field.", "The house stood in the field."),
    ("The house stood in the field .", "The house stood in the field."),
    ("It's a pity, he said, that you're late.", "It 's a pity , he said , that you 're late ."),
    ("It 's a pity , he said , that you 're late .", "It 's a pity , he said , that you 're late ."),
    ("He didn't go (at least not then).", "He did n't go ( at least not then ) ."),
    ('He said "No" twice.', 'He said " No " twice .'),
    ("it’s a change, isn’t it", "it's a change , isn't it"),
    ("“Quite so,” she said.", '" Quite so , " she said .'),
    ("a well-known writer", "a well known writer"),
    ("a well - known writer", "a well-known writer"),
    ("The MP spoke to the BBC.", "The M P spoke to the B B C ."),
    ("Marks and Spencer", "Marks & Spencer"),
    ("the ___ cat *sat* #", "the cat sat"),
    ("They paid £4,000 in 1775.", "They paid £400,000 in 1777 ."),
    ("The prieſts read the leſſon.", "The priests read the lesson."),
    ("café society", "café society"),          # NFD vs NFC: the same text
    ("cafe society", "café society"),                 # a dropped accent is an error
    ("zu fur", "zuͤ fuͤr"),                      # combining e above, final and medial
    ("STRASSE und Straße", "Straße und Straße"),
    ("dragged on ,with him", "dragged on , with him"),
    ("return... ...which shows", "return ... ... which shows"),
    ("He replied: yes; no! perhaps?", "He replied : yes ; no ! perhaps ?"),
    ("Tbe hxvse ls ar al", "The house stood in the field that day"),
    ("one two three four five six seven", "one two three"),
    # edge classes: mark on trailing punctuation, apostrophe kinds at token edges,
    # numbers that are not decimal digits, dashes inside tokens, transposed words
    ("t̶a̶x̶i̶.̶ now", "taxi . now"),
    ("ʼtis the boysʼ donʼt", "'tis the boys' don't"),
    ("12 ½ pounds, he said⁴", "12 1/2 pounds , he said"),
    ("yes—no in 1914–18", "yes-no in 1914-18"),
    ("b a c", "a b c"),
    ("T̈ ÉTÉ ÜBER", "ẗ été über"),
    ('(he cried "stop"!) twice', '( he cried " stop " ! ) twice'),
    ("He said “No” twice.", 'He said " No " twice .'),
    ("a .)a bearing ),and this", "a . ) a bearing ) , and this"),
    ("IT'S HERE, DIDN'T YOU SEE", "IT 'S HERE , DID N'T YOU SEE"),
    ("she said 'so what' again", "she said ' so what ' again"),
    ("word  \n next\t one", "word next one"),
]


def main() -> None:
    golden = []
    for predicted, reference in PAIRS:
        entry = {"predicted": predicted, "reference": reference, "counts": {}}
        for convention in CONVENTIONS:
            result = error_rates(predicted, reference, convention)
            entry["counts"][convention or "none"] = {
                basis: list(getattr(result, basis).model_dump().values()) for basis in BASES
            }
        golden.append(entry)
    payload = {"definition": DEFINITION, "order": ["char_edits", "ref_chars", "word_edits", "ref_words"],
               "pairs": golden}
    GOLDEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN_PATH.name}: {len(golden)} pairs x {len(CONVENTIONS)} conventions")


if __name__ == "__main__":
    main()
