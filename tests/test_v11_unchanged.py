"""Regression oracle: scorer_version="1.1" behavior must never change.

Replays tests/fixtures/v11_pairs.json and compares full reports against
tests/fixtures/v11_snapshot.json. scorer_version "1.1" (strict) is the
version the submitted paper was scored with; any v1.2 feature work that
alters these outputs is a regression by definition. See
tests/make_v11_snapshot.py for what exactly the snapshot pins.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from misnomer import score
from misnomer.config import ScorerConfig

FIXTURES = Path(__file__).parent / "fixtures"


def _assert_close(got, want, path="$"):
    """Recursive equality; floats compared with tolerance (the snapshot
    stores six decimals)."""
    if isinstance(want, float) and isinstance(got, (int, float)):
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-5), f"{path}: {got!r} != {want!r}"
        return
    assert type(got) == type(want), f"{path}: type {type(got).__name__} vs {type(want).__name__}"
    if isinstance(want, dict):
        assert set(got) == set(want), f"{path}: keys differ: {set(got) ^ set(want)}"
        for k in want:
            _assert_close(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert len(got) == len(want), f"{path}: length {len(got)} vs {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_close(g, w, f"{path}[{i}]")
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def _load():
    pairs = json.loads((FIXTURES / "v11_pairs.json").read_text(encoding="utf-8"))
    snapshot = json.loads((FIXTURES / "v11_snapshot.json").read_text(encoding="utf-8"))
    assert len(pairs) == len(snapshot)
    return list(zip(pairs, snapshot))


def test_the_oracle_covers_every_class_and_two_traditions():
    snapshot = [expected for _, expected in _load()]
    classes = {w["error_type"] for r in snapshot for w in r["word_scores"] if w["error_type"]}
    assert classes == {"SEMANTIC", "OBVIOUS", "NUMERIC", "NORMALIZATION", "ARCHAIZING", "GARBLE",
                       "HOMOGLYPH", "SEGMENTATION"}
    assert {r["tradition"] for r in snapshot} == {"modern", "early-modern-latin-print"}
    assert any(r["is_refusal"] for r in snapshot)
    assert all("error_rates" not in r for r in snapshot)


@pytest.mark.parametrize("pair,expected", _load(), ids=lambda p: str(p)[:40])
def test_v11_report_unchanged(pair, expected):
    if isinstance(pair, dict):  # snapshot entry passed as pair — guard mixup
        pytest.skip("fixture ordering mismatch")
    predicted, ground_truth, *overrides = pair
    cfg = ScorerConfig(scorer_version="1.1", strict=True, **(overrides[0] if overrides else {}))
    report = score(predicted, ground_truth, config=cfg)
    _assert_close(report.model_dump(), expected)
