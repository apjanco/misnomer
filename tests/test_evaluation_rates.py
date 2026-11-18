"""The evaluation layer's error-rate tables, checked against the raw
outputs in evaluation/. Skipped when those data files are not present
(they are stored with Git LFS). The rules themselves are tested on
synthetic input in tests/test_evaluation_rates_unit.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).parent.parent / "evaluation"
sys.path.insert(0, str(EVAL))

_probe = EVAL / "claude-vision_results.jsonl"
pytestmark = pytest.mark.skipif(
    not _probe.exists() or _probe.stat().st_size < 10_000,
    reason="evaluation data not checked out",
)

import analysis as A  # noqa: E402
from misnomer.conventions import resolve_convention  # noqa: E402

PINNED = json.loads(
    (Path(__file__).parent / "fixtures" / "iam_error_rates_pinned.json").read_text(encoding="utf-8")
)
# the Lines column of the submitted table, written out here as well so that a
# change to the constant in analysis.py cannot move it silently
PUBLISHED_LINES = {
    "tesseract": 9460, "kraken-mccatmus": 10372, "pylaia-iam": 2915, "pylaia-iam-lm": 2915,
    "chatgpt-gpt4o": 10168, "chatgpt5": 10348, "chandra": 10369, "Qwen3-VL-8B-Instruct": 10370,
    "claude-vision": 10358, "gemini-portkey": 10295,
}


@pytest.fixture(scope="module")
def raw():
    return {name: A.load_raw(name) for name in A.SYSTEMS}


@pytest.fixture(scope="module")
def tables():
    return {"own_line_set": A.error_rate_rows(n_boot=0),
            "test_split": A.error_rate_rows(split="test", n_boot=0)}


def test_references_are_identical_across_systems(raw):
    first = raw["claude-vision"][0]
    for name, (lines, _) in raw.items():
        assert all(lines[i].gt == first[i].gt for i in range(A.N_LINES)), name


def test_loader_agrees_with_the_legacy_loader_when_nothing_is_merged():
    for name, (stem, _) in A.SYSTEMS.items():
        lines, _ = A.load_raw(name, merge_backfill=False)
        legacy = A._raw_meta(stem)
        assert set(legacy) == set(lines)
        for i, (gt_tokens, pred, split) in legacy.items():
            assert (gt_tokens, pred, split) == (lines[i].gt.split(), lines[i].pred, lines[i].split), (name, i)


def test_second_pass_only_fills_lines_left_empty(raw):
    merged, stats = raw["chatgpt5"]
    first, _ = A.load_raw("chatgpt5", merge_backfill=False)
    filled = [i for i in merged if merged[i].backfilled]
    assert stats["backfilled_rows"] == len(filled) == 2264
    assert all(not first[i].pred.strip() and merged[i].pred.strip() for i in filled)
    assert all(merged[i].pred == first[i].pred for i in merged if not merged[i].backfilled)


def test_duplicates_are_resolved_last_write_wins(raw):
    assert raw["gemini-portkey"][1] == {"raw_rows": 10458, "duplicated_indices": 85, "backfilled_rows": 0}
    assert raw["chatgpt5"][1] == {"raw_rows": 10411, "duplicated_indices": 38, "backfilled_rows": 2264}


def test_split_boundaries():
    splits = A.split_map()
    assert [splits[i] for i in (0, 6481, 6482, 7457, 7458, 10372)] == [
        "train", "train", "validation", "validation", "test", "test"]
    assert {s: sum(1 for v in splits.values() if v == s) for s in ("train", "validation", "test")} == {
        "train": 6482, "validation": 976, "test": 2915}
    assert all(splits[i] == "test" for i in range(7458, 10373))


def test_answered_lines_are_the_published_lines_column(tables):
    assert A.PUBLISHED_LINES == PUBLISHED_LINES
    assert {r["system"]: r["views"]["answered"]["lines"] for r in tables["own_line_set"]} == PUBLISHED_LINES


def test_every_figure_of_both_tables_is_pinned(tables):
    assert PINNED["definition"] == "error-rates/1"
    assert PINNED["convention"] == resolve_convention(A.REFERENCE_CONVENTION).identity
    order = PINNED["order"]
    for table, rows in tables.items():
        expected = PINNED["tables"][table]
        assert [r["system"] for r in rows] == list(expected)
        for r in rows:
            want = expected[r["system"]]
            assert r["convention"] == PINNED["convention"] and r["definition"] == PINNED["definition"]
            assert (r["line_set"], r["lines"]) == (want["line_set"], want["lines"]), r["system"]
            assert dict(sorted(r["status"].items())) == want["status"], r["system"]
            assert r["meta_output_lines"] == want["meta_output_lines"]
            assert r["backfilled_lines"] == want["backfilled_lines"]
            assert set(r["views"]) == set(want["views"]) == set(A.VIEWS)
            for view, entry in r["views"].items():
                assert entry["lines"] == want["views"][view]["lines"], (table, r["system"], view)
                for basis in A.ERROR_RATE_BASES:
                    got = [entry[basis][k] for k in order]
                    assert got == want["views"][view][basis], (table, r["system"], view, basis)
                    assert entry[basis]["cer"] == got[0] / got[1] and entry[basis]["wer"] == got[2] / got[3]


def test_headline_figures(tables):
    # content basis, own line set, per cent to two decimals: the figures quoted in the table
    rows = {r["system"]: r["views"] for r in tables["own_line_set"]}
    pct = lambda view, name: (round(100 * rows[name][view]["content"]["cer"], 2),
                              round(100 * rows[name][view]["content"]["wer"], 2))
    assert pct("answered", "claude-vision") == (1.25, 2.99)
    assert pct("all_lines", "claude-vision") == (1.36, 3.08)
    assert pct("answered", "gemini-portkey") == (1.44, 3.26)
    assert pct("answered_excl_meta_output", "gemini-portkey") == (0.97, 2.54)
    assert pct("answered", "pylaia-iam") == (6.85, 23.44)
    assert pct("answered", "pylaia-iam-lm") == (6.60, 21.99)
    assert pct("answered", "tesseract") == (46.21, 92.37)
    assert pct("all_lines", "chatgpt-gpt4o") == (4.75, 8.10)


def test_meta_output_lines_are_answered_lines(raw):
    for name, indices in A.META_OUTPUT.items():
        assert len(set(indices)) == len(indices)
        lines, _ = raw[name]
        assert all(A.line_status(lines[i].pred, lines[i].gt) == "ok" for i in indices), name
    assert (len(A.META_OUTPUT["gemini-portkey"]), len(A.META_OUTPUT["chatgpt-gpt4o"])) == (40, 4)


def test_views_differ_only_where_they_should(tables):
    for r in tables["own_line_set"]:
        v, status = r["views"], r["status"]
        failed = sum(status.get(k, 0) for k in A.FAILED)
        assert v["answered"]["lines"] == status.get("ok", 0)
        assert v["all_lines"]["lines"] == r["lines"] == sum(status.values())
        assert v["answered_excl_meta_output"]["lines"] == v["answered"]["lines"] - r["meta_output_lines"]
        if not r["meta_output_lines"]:
            assert v["answered_excl_meta_output"] == v["answered"]
            assert v["all_lines_meta_output_failed"] == v["all_lines"]
        if failed == 0 and status.get("refusal-like-reference", 0) == 0:
            assert v["answered"] == v["all_lines"]
        for basis in A.ERROR_RATE_BASES:
            # every line adds its reference to the all-lines view (a failed line whose
            # reference holds no word, such as a row of dashes, adds nothing on the content basis)
            assert v["all_lines"][basis]["ref_chars"] >= v["answered"][basis]["ref_chars"]
            assert v["all_lines"][basis]["ref_words"] >= v["answered"][basis]["ref_words"]
        if failed:
            assert v["all_lines"]["verbatim"]["ref_chars"] > v["answered"]["verbatim"]["ref_chars"]


def test_test_split_table_scores_everyone_on_the_same_lines(tables):
    assert {r["lines"] for r in tables["test_split"]} == {2915}
    own = {r["system"]: r for r in tables["own_line_set"]}
    for r in tables["test_split"]:  # PyLaia's own line set IS the test split
        if r["system"].startswith("pylaia"):
            assert r["views"] == own[r["system"]]["views"]


def test_on_this_corpus_no_pooled_word_rate_is_below_its_character_rate(tables):
    # an observation about these ten systems, not a property of the definition
    for rows in tables.values():
        for r in rows:
            for view in A.VIEWS:
                for basis in A.ERROR_RATE_BASES:
                    entry = r["views"][view][basis]
                    assert entry["wer"] >= entry["cer"], (r["system"], view, basis)


def test_convention_is_idempotent_and_whitespace_only_on_all_data(raw):
    conv = resolve_convention(A.REFERENCE_CONVENTION)
    squeeze = lambda s: "".join(s.split())
    for name, (lines, _) in raw.items():
        for ln in lines.values():
            for text in (ln.gt, ln.pred):
                once = conv.apply(text)
                assert conv.apply(once) == once, (name, ln.index)
                assert squeeze(once) == squeeze(text), (name, ln.index)


def test_every_split_clitic_in_the_reference_is_rejoined(raw):
    conv = resolve_convention(A.REFERENCE_CONVENTION)
    lines, _ = raw["claude-vision"]
    clitics = {"'s", "'ll", "'re", "'d", "'m", "'ve"}
    before = sum(1 for ln in lines.values() for t in ln.gt.split() if t.lower() in clitics)
    after = sum(1 for ln in lines.values() for t in conv.apply(ln.gt).split() if t.lower() in clitics)
    assert before == 252 and after == 0
