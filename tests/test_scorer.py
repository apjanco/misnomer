from misnomer.scorer import score, score_batch
from misnomer.config import ScorerConfig


def test_score_generates_report() -> None:
    report = score("The horse stood in field", "The house stood in field")
    assert report.wer > 0
    assert report.document_score >= 0
    assert len(report.word_scores) == 5


def test_score_match_has_zero_document_score() -> None:
    report = score("a b c", "a b c")
    assert report.wer == 0.0
    assert report.document_score == 0.0


def test_fallback_does_not_misclassify_obvious_error() -> None:
    """horse→house is character-similar but semantically distant.
    In fallback (text_only) mode the scorer must NOT classify it as SEMANTIC."""
    cfg = ScorerConfig(
        lm_model="nonexistent/model-does-not-exist",
        embedder_model="nonexistent/embedder-does-not-exist",
        local_files_only=True,
        allow_download=False,
    )
    report = score("The horse stood in the field", "The house stood in the field", config=cfg)
    assert report.scorer_mode in ("text_only", "standard")
    sub = next(ws for ws in report.word_scores if ws.alignment_type == "SUBSTITUTION")
    # Fallback mode: no embedder → error_type must be None, not "SEMANTIC"
    assert sub.error_type is None
    # Unclassified substitutions must not inflate either aggregate counter.
    assert report.semantic_error_count == 0
    assert report.obvious_error_count == 0


def test_fallback_embedding_similarity_is_none() -> None:
    """When the embedder is unavailable, embedding_similarity must be None."""
    cfg = ScorerConfig(
        lm_model="nonexistent/model-does-not-exist",
        embedder_model="nonexistent/embedder-does-not-exist",
        local_files_only=True,
        allow_download=False,
    )
    report = score("The horse stood", "The house stood", config=cfg)
    sub = next(ws for ws in report.word_scores if ws.alignment_type == "SUBSTITUTION")
    assert sub.embedding_similarity is None


def test_report_lm_model_reflects_actual_backend() -> None:
    """lm_model in the report must name what was actually used, not the config value."""
    cfg = ScorerConfig(
        lm_model="nonexistent/model-does-not-exist",
        embedder_model="nonexistent/embedder-does-not-exist",
        local_files_only=True,
        allow_download=False,
    )
    report = score("a b", "a c", config=cfg)
    # In offline fallback, the proxy was used — not the configured transformer name.
    assert report.lm_model == "frequency-proxy"
    assert report.embedder_model == "rapidfuzz-char-similarity"


def test_score_batch_produces_independent_reports() -> None:
    pairs = [
        ("The horse stood", "The house stood"),
        ("a b c", "a b c"),
    ]
    reports = score_batch(pairs)
    assert len(reports) == 2
    assert reports[0].wer > 0
    assert reports[1].wer == 0.0


def test_score_batch_per_pair_metadata() -> None:
    pairs = [
        ("The horse stood", "The house stood", {"page": 1}),
        ("a b c", "a b c", {"page": 2}),
    ]
    reports = score_batch(pairs)  # type: ignore[arg-type]
    assert reports[0].metadata == {"page": 1}
    assert reports[1].metadata == {"page": 2}


def test_perplexity_length_independent_in_proxy() -> None:
    """Proxy perplexity of two unique words of different lengths should be equal
    (same document frequency = same proxy score, no char-length penalty)."""
    cfg = ScorerConfig(
        lm_model="nonexistent/model-does-not-exist",
        local_files_only=True,
        allow_download=False,
    )
    from misnomer.models.lm import LMScorer
    lm = LMScorer(cfg)
    # Two unique words: each appears once, so same rarity → same proxy perplexity.
    perplexities = lm.word_perplexities(["cat", "elephant"])
    assert perplexities[0] == perplexities[1]


import pytest

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch")
    or not __import__("importlib").util.find_spec("sentence_transformers"),
    reason="requires torch and sentence-transformers",
)
def test_spec_examples_full_mode() -> None:
    """Core spec claim: horse→steed (semantic) must score higher than horse→house (obvious)
    and be classified correctly in full mode."""
    cfg = ScorerConfig(local_files_only=False, allow_download=True)
    r_steed = score("The steed stood in the field", "The horse stood in the field", config=cfg)
    r_house = score("The house stood in the field", "The horse stood in the field", config=cfg)

    assert r_steed.scorer_mode == "full", "Test requires full scorer mode"

    sub_steed = next(ws for ws in r_steed.word_scores if ws.alignment_type == "SUBSTITUTION")
    sub_house = next(ws for ws in r_house.word_scores if ws.alignment_type == "SUBSTITUTION")

    assert sub_steed.error_type == "SEMANTIC", f"horse→steed should be SEMANTIC, got {sub_steed.error_type} (sim={sub_steed.embedding_similarity:.3f})"
    assert sub_house.error_type == "OBVIOUS", f"horse→house should be OBVIOUS, got {sub_house.error_type} (sim={sub_house.embedding_similarity:.3f})"
    assert sub_steed.composite_score > sub_house.composite_score, (
        f"horse→steed composite ({sub_steed.composite_score:.3f}) must exceed "
        f"horse→house composite ({sub_house.composite_score:.3f})"
    )

