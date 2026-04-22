from misnomer.scorer import score


def test_score_generates_report() -> None:
    report = score("The horse stood in field", "The house stood in field")
    assert report.wer > 0
    assert report.document_score >= 0
    assert len(report.word_scores) == 5


def test_score_match_has_zero_document_score() -> None:
    report = score("a b c", "a b c")
    assert report.wer == 0.0
    assert report.document_score == 0.0

