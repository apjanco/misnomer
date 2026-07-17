"""Δ = log P(ŵ|c) − log P(w*|c): full-word, teacher-forced, prefix-shared.

Test corpus examples appear here and only here (never in src/): the
mechanisms under test are fully general.
"""

from __future__ import annotations

import pytest

from misnomer.config import ScorerConfig
from misnomer.models.lm import LMScorer


@pytest.fixture(scope="module")
def lm() -> LMScorer:
    scorer = LMScorer(ScorerConfig())
    assert scorer.is_transformer_backed, "cached models required (run P0 first)"
    return scorer


def test_vinegar_pair_delta_positive(lm):
    gt = ["The", "parable", "of", "the", "vinegar"]
    deltas = lm.substitution_deltas(gt, {4: "vineyard"})
    assert deltas[4] is not None
    assert deltas[4] > 0, f"prior should prefer the expected reading, got Δ={deltas[4]:.3f}"


def test_shared_first_token_pair_is_nonzero(lm):
    """First-token-only Δ would collapse to zero when candidates share their
    opening subword; full-word scoring must not."""
    tok = lm.tokenizer
    candidates = [
        ("vineyard", "vinegar"),
        ("unhappily", "unhappiness"),
        ("prevent", "preview"),
        ("relearn", "relight"),
    ]
    pair = None
    for a, b in candidates:
        ta = tok(" " + a, add_special_tokens=False)["input_ids"]
        tb = tok(" " + b, add_special_tokens=False)["input_ids"]
        if len(ta) > 1 and len(tb) > 1 and ta[0] == tb[0]:
            pair = (a, b)
            break
    if pair is None:
        pytest.skip("no shared-first-token pair under this tokenizer")
    a, b = pair
    gt = ["She", "mentioned", "the", b, "again"]
    deltas = lm.substitution_deltas(gt, {3: a})
    assert deltas[3] is not None
    assert deltas[3] != 0.0, f"shared-first-token pair {pair} produced Δ=0"


def test_antisymmetry(lm):
    gt_b = ["The", "horse", "stood", "in", "the", "field"]
    gt_a = ["The", "steed", "stood", "in", "the", "field"]
    d_ab = lm.substitution_deltas(gt_b, {1: "steed"})[1]  # wrote steed, page says horse
    d_ba = lm.substitution_deltas(gt_a, {1: "horse"})[1]  # wrote horse, page says steed
    assert d_ab is not None and d_ba is not None
    assert d_ab == pytest.approx(-d_ba, abs=1e-4)


def test_line_initial_substitution_scores(lm):
    gt = ["Founded", "in", "the", "spring"]
    deltas = lm.substitution_deltas(gt, {0: "Sounded"})
    assert deltas[0] is not None  # BOS/EOS anchor, degenerates toward frequency contrast


def test_determinism(lm):
    gt = ["The", "order", "of", "the", "garter"]
    first = lm.substitution_deltas(gt, {1: "ordre"})
    second = lm.substitution_deltas(gt, {1: "ordre"})
    assert first == second


def test_batch_matches_individual(lm):
    gt = ["The", "small", "hound", "slept", "near", "the", "door"]
    subs = {1: "large", 6: "gate"}
    batched = lm.substitution_deltas(gt, subs)
    single_1 = lm.substitution_deltas(gt, {1: "large"})[1]
    single_6 = lm.substitution_deltas(gt, {6: "gate"})[6]
    assert batched[1] == pytest.approx(single_1, abs=1e-4)
    assert batched[6] == pytest.approx(single_6, abs=1e-4)


def test_degenerate_cases(lm):
    gt = ["a", "plain", "line"]
    assert lm.substitution_deltas(gt, {}) == {}
    assert lm.substitution_deltas(gt, {1: "plain"})[1] == 0.0  # identical pair
    assert lm.substitution_deltas(gt, {1: "  "})[1] is None  # blank candidate


def test_proxy_mode_returns_none():
    cfg = ScorerConfig(
        lm_model="nonexistent/misnomer-test-lm",
        local_files_only=True,
        allow_download=False,
    )
    lm = LMScorer(cfg)
    assert not lm.is_transformer_backed
    assert lm.substitution_deltas(["a", "b"], {1: "c"}) == {1: None}


def test_invalid_dtype_raises_not_degrades():
    with pytest.raises(ValueError, match="torch_dtype"):
        LMScorer(ScorerConfig(torch_dtype="float8"))


def test_device_resolution():
    import torch

    assert LMScorer._resolve_device(torch, "cpu") == "cpu"
    assert LMScorer._resolve_device(torch, "auto") in {"mps", "cuda", "cpu"}
