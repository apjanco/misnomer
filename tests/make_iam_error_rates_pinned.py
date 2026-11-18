"""Pin the IAM error-rate tables as integers.

    uv run python tests/make_iam_error_rates_pinned.py

Writes tests/fixtures/iam_error_rates_pinned.json: for both tables (each
system's own line set; the test split for every system), per system, the
line counts by status and the four integers of every view and basis.
tests/test_evaluation_rates.py recomputes them from the raw outputs and
asserts equality, so no figure of the table can move unnoticed.

The values written on 2026-10-01 were confirmed line by line and in total by
an independent implementation of the definition. Regenerate only when the
definition, the convention, the view rules or the data change on purpose —
and say so where the figures are quoted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL = Path(__file__).parent.parent / "evaluation"
sys.path.insert(0, str(EVAL))

import analysis as A  # noqa: E402

PINNED_PATH = Path(__file__).parent / "fixtures" / "iam_error_rates_pinned.json"
ORDER = ["char_edits", "ref_chars", "word_edits", "ref_words"]


def pinned_tables() -> dict:
    tables = {}
    for table, split in (("own_line_set", None), ("test_split", "test")):
        systems = {}
        for row in A.error_rate_rows(split=split, n_boot=0):
            systems[row["system"]] = {
                "line_set": row["line_set"],
                "lines": row["lines"],
                "status": dict(sorted(row["status"].items())),
                "meta_output_lines": row["meta_output_lines"],
                "backfilled_lines": row["backfilled_lines"],
                "views": {
                    view: {"lines": entry["lines"],
                           **{basis: [entry[basis][k] for k in ORDER] for basis in A.ERROR_RATE_BASES}}
                    for view, entry in row["views"].items()
                },
            }
            convention, definition = row["convention"], row["definition"]
        tables[table] = systems
    return {"definition": definition, "convention": convention, "order": ORDER, "tables": tables}


def main() -> None:
    PINNED_PATH.write_text(json.dumps(pinned_tables(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {PINNED_PATH.name}")


if __name__ == "__main__":
    main()
