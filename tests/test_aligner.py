from misnomer.aligner import align_words


def test_align_words_substitution() -> None:
    aligned = align_words("The horse stood", "The house stood")
    assert [x.alignment_type for x in aligned] == ["MATCH", "SUBSTITUTION", "MATCH"]
    assert aligned[1].predicted_word == "horse"
    assert aligned[1].ground_truth_word == "house"

