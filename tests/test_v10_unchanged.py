"""Regression oracle: scorer_version="1.0" behavior must never change.

Replays tests/fixtures/v10_pairs.json and compares full reports against
tests/fixtures/v10_snapshot.json (floats to 1e-4). Any v1.1 feature work
that alters these outputs is a regression by definition.
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
    """Recursive equality; floats compared with tolerance (snapshot stores 6dp,
    so worst-case storage error is 5e-7 — never round-then-compare)."""
    if isinstance(want, float) and isinstance(got, (int, float)):
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-5), (
            f"{path}: {got!r} != {want!r}"
        )
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
    pairs = json.loads((FIXTURES / "v10_pairs.json").read_text())
    snapshot = json.loads((FIXTURES / "v10_snapshot.json").read_text())
    return list(zip(pairs, snapshot))


@pytest.fixture(scope="module")
def cfg():
    return ScorerConfig(scorer_version="1.0")


@pytest.mark.parametrize("pair,expected", _load(), ids=lambda p: str(p)[:40])
def test_v10_report_unchanged(pair, expected, cfg):
    if isinstance(pair, dict):  # snapshot entry passed as pair — guard mixup
        pytest.skip("fixture ordering mismatch")
    predicted, ground_truth = pair
    report = score(predicted, ground_truth, config=cfg)
    _assert_close(report.model_dump(), expected)
