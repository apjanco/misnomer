"""The evaluation layer's error-rate code on small synthetic inputs.

Needs no evaluation data: every test builds its own result files in a
temporary directory, so the rules of the views are pinned exactly even when
the corpus is not checked out.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).parent.parent / "evaluation"
sys.path.insert(0, str(EVAL))

import analysis as A  # noqa: E402
from misnomer.rates import error_rates  # noqa: E402

REFUSAL = "I'm sorry, I cannot transcribe this image."
# index: (split, reference, {system: prediction})
LINES = [
    ("train", "the house stood in the field", {"sys": "the house stood in the field"}),          # ok, exact
    ("train", "a second line of text", {"sys": ""}),                                               # empty
    ("train", "a third line of text", {"sys": REFUSAL}),                                           # refusal
    ("train", "short", {"sys": "x" * 40}),                                                         # over-long
    ("test", "I can't recall a thing", {"sys": "I can't recall a thing"}),                         # refusal-like reference
    ("test", "the horse stood , he said", {"sys": "the house stood, he said"}),                    # ok, one word wrong
    ("test", "words on the last line", {"sys": "Let me look again. The word is `last`"}),          # meta-output
    ("test", "it 's fine", {"sys": "it's fine"}),                                                  # ok, layout only
]


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    def write(stem, rows):
        with open(tmp_path / f"{stem}_results.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    for stem in ("pylaia-iam", "pylaia-iam-lm"):
        write(stem, [{"index": i, "ground_truth": gt, "predicted_text": gt, "split": split}
                     for i, (split, gt, _) in enumerate(LINES)])
    rows = [{"index": i, "ground_truth": gt, "predicted_text": preds["sys"]}
            for i, (_, gt, preds) in enumerate(LINES)]
    rows.insert(1, {"index": 0, "ground_truth": LINES[0][1], "predicted_text": ""})  # a failed try, then the retry wins
    rows.append(rows.pop(1))
    rows.append({"index": 0, "ground_truth": LINES[0][1], "predicted_text": LINES[0][2]["sys"]})
    write("sys", rows)

    monkeypatch.setattr(A, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(A, "N_LINES", len(LINES))
    monkeypatch.setattr(A, "SYSTEMS", {"sys": ("sys", False), "pylaia-iam": ("pylaia-iam", True),
                                       "pylaia-iam-lm": ("pylaia-iam-lm", True)})
    monkeypatch.setattr(A, "PUBLISHED_LINES", {"sys": 4, "pylaia-iam": 3, "pylaia-iam-lm": 3})
    monkeypatch.setattr(A, "BACKFILL", {})
    monkeypatch.setattr(A, "META_OUTPUT", {"sys": (6,)})
    A.split_reference.cache_clear()
    yield tmp_path
    A.split_reference.cache_clear()


def flat(entry: dict, basis: str) -> tuple[int, int, int, int]:
    return tuple(entry[basis][k] for k in ("char_edits", "ref_chars", "word_edits", "ref_words"))


def summed(pairs: list[tuple[str, str]], basis: str) -> tuple[int, int, int, int]:
    totals = [0, 0, 0, 0]
    for pred, gt in pairs:
        counts = getattr(error_rates(pred, gt, A.REFERENCE_CONVENTION), basis)
        for k, value in enumerate((counts.char_edits, counts.ref_chars, counts.word_edits, counts.ref_words)):
            totals[k] += value
    return tuple(totals)


def test_line_status_rules():
    assert A.line_status("", "some reference text") == "empty"
    assert A.line_status("   \n", "some reference text") == "empty"
    assert A.line_status(REFUSAL, "the reference") == "refusal"
    assert A.line_status("I can't recall a thing", "I can't recall a thing") == "refusal-like-reference"
    assert A.line_status("x" * 61, "y" * 20) == "degenerate"       # 61 > 2*20 + 20
    assert A.line_status("x" * 60, "y" * 20) == "ok"


def test_duplicates_resolve_to_the_last_row(synthetic):
    lines, stats = A.load_raw("sys")
    assert stats == {"raw_rows": 10, "duplicated_indices": 1, "backfilled_rows": 0}
    assert lines[0].pred == "the house stood in the field"


def test_views_follow_their_rules_exactly(synthetic):
    row = {r["system"]: r for r in A.error_rate_rows(n_boot=0)}["sys"]
    assert row["lines"] == 8 and row["line_set"] == "all"
    assert row["status"] == {"ok": 4, "empty": 1, "refusal": 1, "degenerate": 1, "refusal-like-reference": 1}
    assert row["meta_output_lines"] == 1 and row["backfilled_lines"] == 0
    gt = [g for _, g, _ in LINES]
    pred = [p["sys"] for _, _, p in LINES]
    answered = [(pred[i], gt[i]) for i in (0, 5, 6, 7)]
    all_lines = [(pred[0], gt[0]), ("", gt[1]), ("", gt[2]), ("", gt[3]),   # failures: scored as deleted
                 (pred[4], gt[4]),                                           # refusal-like reference: scored as given
                 (pred[5], gt[5]), (pred[6], gt[6]), (pred[7], gt[7])]
    answered_excl = [(pred[i], gt[i]) for i in (0, 5, 7)]
    all_meta_failed = all_lines[:6] + [("", gt[6]), all_lines[7]]
    expected = {"answered": answered, "all_lines": all_lines,
                "answered_excl_meta_output": answered_excl,
                "all_lines_meta_output_failed": all_meta_failed}
    assert set(row["views"]) == set(A.VIEWS) == set(expected)
    for view, pairs in expected.items():
        assert row["views"][view]["lines"] == len(pairs), view
        for basis in A.ERROR_RATE_BASES:
            assert flat(row["views"][view], basis) == summed(pairs, basis), (view, basis)
    # spelled out by hand for the content basis of the answered lines without the meta-output line:
    # line 0 exact (0 of 28 characters, 0 of 6 words); line 5 "house" for "horse" (1 of 23, 1 of 5);
    # line 7 differs in layout only (0 of 9, 0 of 2)
    assert flat(row["views"]["answered_excl_meta_output"], "content") == (1, 60, 1, 13)
    # the meta-output line adds its 22 reference characters and 5 reference words to the answered view
    assert row["views"]["answered"]["content"]["ref_chars"] == 82
    assert row["views"]["answered"]["content"]["ref_words"] == 18
    assert row["views"]["answered"]["content"]["char_edits"] > 20
    for view in A.VIEWS:
        for basis in A.ERROR_RATE_BASES:
            e = row["views"][view][basis]
            assert e["cer"] == e["char_edits"] / e["ref_chars"] and e["wer"] == e["word_edits"] / e["ref_words"]


def test_a_failed_line_is_scored_as_deleted_not_on_its_text(synthetic):
    row = {r["system"]: r for r in A.error_rate_rows(n_boot=0)}["sys"]
    # the 40-character runaway on the reference "short" costs 5 characters, not 40
    everything = row["views"]["all_lines"]["verbatim"]
    answered = row["views"]["answered"]["verbatim"]
    gt = [g for _, g, _ in LINES]
    failed_and_like = (everything["char_edits"] - answered["char_edits"],
                       everything["ref_chars"] - answered["ref_chars"])
    assert failed_and_like == (len(gt[1]) + len(gt[2]) + len(gt[3]) + 0,
                               len(gt[1]) + len(gt[2]) + len(gt[3]) + len(gt[4]))


def test_test_split_restricts_every_system(synthetic):
    rows = {r["system"]: r for r in A.error_rate_rows(split="test", n_boot=0)}
    assert {r["lines"] for r in rows.values()} == {4}
    assert rows["sys"]["status"] == {"refusal-like-reference": 1, "ok": 3}
    own = {r["system"]: r for r in A.error_rate_rows(n_boot=0)}
    assert own["pylaia-iam"]["line_set"] == "test" and own["pylaia-iam"]["views"] == rows["pylaia-iam"]["views"]


def test_other_splits_are_refused(synthetic):
    with pytest.raises(ValueError, match="split must be"):
        A.error_rate_rows(split="train", n_boot=0)


def test_convention_none_compares_strings_as_supplied(synthetic):
    row = {r["system"]: r for r in A.error_rate_rows(convention=None, n_boot=0)}["sys"]
    assert row["convention"] is None
    with_convention = {r["system"]: r for r in A.error_rate_rows(n_boot=0)}["sys"]
    assert with_convention["convention"]["name"] == "tokenised-english"
    # "it 's fine" vs "it's fine" costs words only without the convention
    assert row["views"]["answered"]["content"]["word_edits"] > with_convention["views"]["answered"]["content"]["word_edits"]


def test_inconsistent_inputs_are_errors(synthetic):
    path = synthetic / "sys_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # a reference that differs from the split source
    changed = [dict(r, ground_truth="another reference") if r["index"] == 3 else r for r in rows]
    path.write_text("\n".join(json.dumps(r) for r in changed) + "\n")
    with pytest.raises(ValueError, match="differs from the split source"):
        A.error_rate_rows(n_boot=0)
    # rows of one index disagreeing on the reference
    path.write_text("\n".join(json.dumps(r) for r in rows + [dict(rows[0], ground_truth="zzz")]) + "\n")
    with pytest.raises(ValueError, match="disagree on the reference"):
        A.load_raw("sys")
    # a retry that would replace text with nothing
    path.write_text("\n".join(json.dumps(r) for r in rows + [dict(rows[2], predicted_text="")]) + "\n")
    with pytest.raises(ValueError, match="would drop text"):
        A.load_raw("sys")
    # a missing index
    path.write_text("\n".join(json.dumps(r) for r in rows if r["index"] != 5) + "\n")
    with pytest.raises(ValueError, match="index coverage"):
        A.load_raw("sys")


def test_meta_output_must_be_answered_lines(synthetic, monkeypatch):
    monkeypatch.setattr(A, "META_OUTPUT", {"sys": (1,)})   # line 1 is empty, not answered
    with pytest.raises(ValueError, match="META_OUTPUT line 1"):
        A.error_rate_rows(n_boot=0)


def test_second_pass_fills_only_empty_lines(synthetic, monkeypatch):
    monkeypatch.setattr(A, "BACKFILL", {"sys": "sys_second.jsonl"})
    second = synthetic / "sys_second.jsonl"
    second.write_text(json.dumps({"index": 1, "ground_truth": LINES[1][1], "predicted_text": "a second line of text"}) + "\n")
    monkeypatch.setattr(A, "PUBLISHED_LINES", {"sys": 5, "pylaia-iam": 3, "pylaia-iam-lm": 3})
    lines, stats = A.load_raw("sys")
    assert stats["backfilled_rows"] == 1 and lines[1].backfilled and lines[1].pred == "a second line of text"
    row = {r["system"]: r for r in A.error_rate_rows(n_boot=0)}["sys"]
    assert row["backfilled_lines"] == 1 and row["views"]["answered"]["lines"] == 5
    assert A.load_raw("sys", merge_backfill=False)[0][1].pred == ""
    second.write_text(json.dumps({"index": 0, "ground_truth": LINES[0][1], "predicted_text": "x"}) + "\n")
    with pytest.raises(ValueError, match="was not empty in the first pass"):
        A.load_raw("sys")
    second.write_text(json.dumps({"index": 1, "ground_truth": "other", "predicted_text": "x"}) + "\n")
    with pytest.raises(ValueError, match="reference differs"):
        A.load_raw("sys")


def test_published_line_count_is_enforced(synthetic, monkeypatch):
    monkeypatch.setattr(A, "PUBLISHED_LINES", {"sys": 5, "pylaia-iam": 3, "pylaia-iam-lm": 3})
    with pytest.raises(AssertionError, match="published table"):
        A.error_rate_rows(n_boot=0)


# --- bootstrap ------------------------------------------------------------

def test_bootstrap_of_identical_lines_is_a_point():
    assert A._bootstrap_rate([1] * 50, [4] * 50, 200, 1) == (0.25, 0.25)


def test_bootstrap_brackets_the_rate_and_is_reproducible():
    edits = [0] * 90 + [10] * 10
    refs = [10] * 100
    lo, hi = A._bootstrap_rate(edits, refs, 2000, 7)
    assert lo < 0.10 < hi and 0.03 < lo < 0.07 and 0.13 < hi < 0.18   # binomial sd is 0.03
    assert A._bootstrap_rate(edits, refs, 2000, 7) == (lo, hi)
    # lines of varying length, so that the interval is not confined to a coarse grid
    varied_edits = [i % 7 for i in range(300)]
    varied_refs = [10 + (i * 37) % 23 for i in range(300)]
    assert A._bootstrap_rate(varied_edits, varied_refs, 500, 7) == A._bootstrap_rate(varied_edits, varied_refs, 500, 7)
    assert A._bootstrap_rate(varied_edits, varied_refs, 500, 7) != A._bootstrap_rate(varied_edits, varied_refs, 500, 8)


def test_bootstrap_is_a_ratio_of_sums_not_a_mean_of_ratios():
    # one short line all wrong, one long line all right: ratio of sums 1/101, mean of ratios 0.5
    lo, hi = A._bootstrap_rate([1, 0], [1, 100], 4000, 3)
    assert lo == 0.0 and hi == 1.0
    assert A._bootstrap_rate([1, 0] * 200, [1, 100] * 200, 500, 3)[1] < 0.02


def test_bootstrap_has_no_interval_without_a_denominator():
    assert A._bootstrap_rate([], [], 100, 1) is None
    assert A._bootstrap_rate([3], [0], 100, 1) is None


def test_intervals_are_attached_when_asked_for(synthetic):
    row = {r["system"]: r for r in A.error_rate_rows(n_boot=50, seed=5)}["sys"]
    entry = row["views"]["answered"]["content"]
    assert entry["cer_ci95"][0] <= entry["cer"] <= entry["cer_ci95"][1]
    assert entry["wer_ci95"][0] <= entry["wer"] <= entry["wer_ci95"][1]
    assert "cer_ci95" not in {r["system"]: r for r in A.error_rate_rows(n_boot=0)}["sys"]["views"]["answered"]["content"]


def test_an_empty_view_has_no_rates(synthetic, monkeypatch):
    monkeypatch.setattr(A, "META_OUTPUT", {})
    entry = A._view([], n_boot=10, seed=1)
    assert entry == {"lines": 0,
                     "verbatim": {"char_edits": 0, "ref_chars": 0, "word_edits": 0, "ref_words": 0, "cer": None, "wer": None},
                     "content": {"char_edits": 0, "ref_chars": 0, "word_edits": 0, "ref_words": 0, "cer": None, "wer": None}}


# --- printing and writing -------------------------------------------------

def test_table_prints_the_right_cells_and_says_what_they_are(synthetic):
    rows = A.error_rate_rows(n_boot=0)
    text = A.error_rate_table(rows)
    sys_row = {r["system"]: r for r in rows}["sys"]
    line = next(l for l in text.splitlines() if l.startswith("sys "))
    a, e = sys_row["views"]["answered"], sys_row["views"]["all_lines"]
    expected = [f"{100 * v:.2f}" for v in (a["content"]["cer"], a["content"]["wer"], a["verbatim"]["cer"],
                                           a["verbatim"]["wer"], e["content"]["cer"], e["content"]["wer"],
                                           e["verbatim"]["cer"], e["verbatim"]["wer"])]
    assert line.replace("|", " ").split() == ["sys", "8", "4"] + expected
    assert "rates in per cent" in text and "tokenised-english" in text and "error-rates/1" in text
    assert "scored as fully deleted" in text and "META_OUTPUT" in text
    sensitivity = next(l for l in text.splitlines() if l.startswith("sys ") and l != line)
    x, f = sys_row["views"]["answered_excl_meta_output"], sys_row["views"]["all_lines_meta_output_failed"]
    assert sensitivity.replace("|", " ").split() == ["sys", "1"] + [
        f"{100 * v:.2f}" for v in (a["content"]["cer"], a["content"]["wer"], x["content"]["cer"], x["content"]["wer"],
                                   e["content"]["cer"], e["content"]["wer"], f["content"]["cer"], f["content"]["wer"])]


def test_written_file_identifies_code_data_and_settings(synthetic, tmp_path):
    import hashlib

    import misnomer.rates

    tables = {"own_line_set": A.error_rate_rows(n_boot=20, seed=11)}
    out = tmp_path / "out" / "rates.json"
    payload = A.write_error_rates(out, tables, n_boot=20, seed=11)
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(payload))
    assert set(stored) >= {"git_commit", "git_dirty", "code_sha256", "versions", "prediction_input",
                           "intervals", "inputs_sha256", "second_pass", "meta_output", "tables"}
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert stored["code_sha256"]["misnomer/rates.py"] == sha(misnomer.rates.__file__)
    assert stored["code_sha256"]["evaluation/analysis.py"] == sha(A.__file__)
    assert set(stored["code_sha256"]) == {"misnomer/rates.py", "misnomer/conventions.py",
                                          "misnomer/data/conventions.json", "evaluation/analysis.py"}
    assert stored["inputs_sha256"]["sys_results.jsonl"] == sha(synthetic / "sys_results.jsonl")
    assert stored["intervals"]["n_boot"] == 20 and stored["intervals"]["seed"] == 11
    assert set(stored["versions"]) == {"python", "unicodedata", "misnomer", "rapidfuzz", "pydantic", "numpy"}
    assert stored["meta_output"] == {"sys": [6]}
    assert stored["tables"]["own_line_set"][0]["convention"]["name"] == "tokenised-english"


def test_legacy_tables_do_not_need_the_package():
    # importing the analysis module must not pull in misnomer (the deposit ships it stand-alone)
    code = ("import sys; sys.path.insert(0, %r); import analysis; "
            "print(any(m == 'misnomer' or m.startswith('misnomer.') for m in sys.modules))" % str(EVAL))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
