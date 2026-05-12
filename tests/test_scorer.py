from misnomer.scorer import score, score_batch, score_jsonl
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


# ---------------------------------------------------------------------------
# New-field tests (Priority 4)
# ---------------------------------------------------------------------------

def test_is_refusal_on_empty_input() -> None:
    """Empty and whitespace-only predictions must be flagged as refusals."""
    for pred in ["", "   ", None]:
        report = score(pred, "The house stood in the field")
        assert report.is_refusal is True, f"Expected is_refusal for {pred!r}"
        assert report.document_error_type == "refusal"
        assert report.semantic_error_count == 0
        assert report.word_scores == []


def test_cer_is_zero_on_exact_match() -> None:
    report = score("exact match text", "exact match text")
    assert report.cer == 0.0


def test_cer_positive_on_char_error() -> None:
    """A single-character substitution must produce CER > 0."""
    report = score("hose", "horse")
    assert report.cer > 0.0


def test_insertion_ratio_on_extra_words() -> None:
    """Prediction with extra words not in GT must produce insertion_ratio > 0."""
    report = score("The horse stood there quietly", "The horse stood")
    assert report.insertion_ratio > 0.0


def test_substitution_rate_on_substitution() -> None:
    report = score("The horse stood", "The house stood")
    assert report.substitution_rate > 0.0


def test_document_error_type_correct_on_exact_match() -> None:
    report = score("a b c", "a b c")
    assert report.document_error_type == "correct"


def test_document_error_type_partial_on_deletion() -> None:
    """A deletion (WER > 0, no substitutions) must yield 'partial', not 'correct'."""
    report = score("The stood", "The horse stood")
    assert report.wer > 0
    assert report.document_error_type == "partial"


def test_document_error_type_partial_on_insertion() -> None:
    """An insertion (WER > 0, no substitutions) must yield 'partial', not 'correct'."""
    report = score("The horse quickly stood", "The horse stood")
    assert report.wer > 0
    assert report.document_error_type == "partial"


def test_public_api_imports() -> None:
    """Top-level misnomer package must export all public symbols."""
    import misnomer
    for name in ["score", "score_batch", "score_jsonl", "highlight",
                 "ScorerConfig", "SemanticErrorReport", "WordScore", "DocumentErrorType"]:
        assert hasattr(misnomer, name), f"misnomer.{name} not found in public API"


def test_score_jsonl_roundtrip(tmp_path) -> None:
    """score_jsonl must score every record and write valid JSON reports."""
    import json
    pairs = [
        {"predicted": "The horse stood", "ground_truth": "The house stood"},
        {"predicted": "a b c", "ground_truth": "a b c"},
        {"predicted": "", "ground_truth": "something"},
    ]
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"
    input_file.write_text("\n".join(json.dumps(p) for p in pairs), encoding="utf-8")

    count = score_jsonl(input_file, output_file)
    assert count == 3

    results = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert len(results) == 3
    assert results[2]["is_refusal"] is True
    assert results[1]["wer"] == 0.0
    assert results[0]["wer"] > 0.0


def test_score_jsonl_passthrough_metadata(tmp_path) -> None:
    """Fields other than predicted/ground_truth must appear in report metadata."""
    import json
    record = {"predicted": "x", "ground_truth": "y", "page": 42, "model": "test"}
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"
    input_file.write_text(json.dumps(record), encoding="utf-8")

    score_jsonl(input_file, output_file)
    result = json.loads(output_file.read_text(encoding="utf-8"))
    assert result["metadata"]["page"] == 42
    assert result["metadata"]["model"] == "test"


def test_score_batch_loads_models_once() -> None:
    """score_batch must produce the same results as calling score() individually."""
    pairs = [
        ("The horse stood", "The house stood"),
        ("a b c", "a b c"),
        ("", "something"),
    ]
    batch_reports = score_batch(pairs)
    individual_reports = [score(p, g) for p, g in pairs]

    for b, i in zip(batch_reports, individual_reports):
        assert b.wer == i.wer
        assert b.is_refusal == i.is_refusal
        assert b.document_error_type == i.document_error_type

