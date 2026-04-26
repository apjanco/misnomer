from misnomer.aligner import align_words, tokenize_words


def test_align_words_substitution() -> None:
    aligned = align_words("The horse stood", "The house stood")
    assert [x.alignment_type for x in aligned] == ["MATCH", "SUBSTITUTION", "MATCH"]
    assert aligned[1].predicted_word == "horse"
    assert aligned[1].ground_truth_word == "house"


def test_tokenize_words_strips_boundary_punctuation() -> None:
    assert tokenize_words("Macleod,") == ["Macleod"]
    assert tokenize_words("Mr.") == ["Mr"]
    assert tokenize_words("chief aide , Mr. Julius") == ["chief", "aide", "Mr", "Julius"]
    # standalone punctuation is discarded entirely
    assert tokenize_words(", .") == []


def test_align_words_punctuation_spacing_ignored() -> None:
    """Space before punctuation (OCR artifact) must not produce substitutions."""
    aligned = align_words(
        "Macleod , is insisting on a policy of change .",
        "Macleod, is insisting on a policy of change.",
    )
    types = [x.alignment_type for x in aligned]
    assert all(t == "MATCH" for t in types), types

