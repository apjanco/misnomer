"""Strict mode: the full model stack or a hard error — never a silent fallback.

Background: without sentence-transformers the embedder degrades to character
similarity, which inverts the SEMANTIC/OBVIOUS taxonomy (see
misnomer-review.md, item 1.1). Strict mode turns that degradation into
StrictModeError. All paper/production runs set strict=True.
"""

from __future__ import annotations

import pytest

from misnomer import score
from misnomer.config import ScorerConfig
from misnomer.scorer import StrictModeError


def _broken_cfg(**overrides) -> ScorerConfig:
    """A config whose models can never load (offline + nonexistent names)."""
    defaults = dict(
        strict=True,
        lm_model="nonexistent/misnomer-test-lm",
        embedder_model="nonexistent/misnomer-test-embedder",
        local_files_only=True,
        allow_download=False,
    )
    defaults.update(overrides)
    return ScorerConfig(**defaults)


def test_strict_raises_when_both_backends_missing():
    with pytest.raises(StrictModeError, match="frequency-proxy"):
        score("a word", "a ward", config=_broken_cfg())


def test_strict_raises_when_only_embedder_missing():
    cfg = _broken_cfg(lm_model="Qwen/Qwen2.5-0.5B")  # LM cached; embedder bogus
    with pytest.raises(StrictModeError, match="char-similarity"):
        score("a word", "a ward", config=cfg)


def test_non_strict_still_degrades_silently():
    cfg = _broken_cfg(strict=False)
    report = score("a word", "a ward", config=cfg)
    assert report.scorer_mode == "text_only"
    assert report.strict is False


def test_strict_full_mode_passes_and_is_reported():
    cfg = ScorerConfig(strict=True)  # models cached by P0
    report = score("The steed stood.", "The horse stood.", config=cfg)
    assert report.scorer_mode == "full"
    assert report.strict is True


def test_strict_applies_to_refusal_path_too():
    # Refusals return early, but a strict run must still guarantee the stack.
    with pytest.raises(StrictModeError):
        score("I'm sorry, I cannot transcribe this image.", "text", config=_broken_cfg())


def test_gates_version_activation():
    assert ScorerConfig(scorer_version="1.0").gates_active is False
    assert ScorerConfig(scorer_version="1.1").gates_active is True
    assert ScorerConfig(scorer_version="1.1-rc").gates_active is True
    assert ScorerConfig(scorer_version="2.0").gates_active is True
    assert ScorerConfig(scorer_version="custom").gates_active is False
