"""Error rates on declared bases (definition "error-rates/1")."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydantic import ValidationError

import misnomer
from misnomer.conventions import resolve_convention
from misnomer.rates import (
    BASES,
    DEFINITION,
    ConventionIdentity,
    ErrorCounts,
    ErrorRates,
    content_view,
    error_rates,
    pool,
    verbatim_view,
)

TOK = "tokenised-english"
GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "error_rates_golden.json").read_text(encoding="utf-8")
)


def counts(result: ErrorRates, basis: str) -> tuple[int, int, int, int]:
    c = getattr(result, basis)
    return (c.char_edits, c.ref_chars, c.word_edits, c.ref_words)


# --- the golden fixture pins the definition -------------------------------

def test_golden_is_for_this_definition():
    assert GOLDEN["definition"] == DEFINITION
    assert GOLDEN["order"] == ["char_edits", "ref_chars", "word_edits", "ref_words"]


@pytest.mark.parametrize("entry", GOLDEN["pairs"], ids=lambda e: e["predicted"][:30] or "<empty>")
@pytest.mark.parametrize("convention", [None, TOK])
def test_golden_counts(entry, convention):
    result = error_rates(entry["predicted"], entry["reference"], convention)
    expected = entry["counts"][convention or "none"]
    for basis in BASES:
        assert list(counts(result, basis)) == expected[basis], basis


# --- both rates come from the same sequence, per basis --------------------

def test_identical_strings_have_no_edits():
    r = error_rates("The house stood in the field.", "The house stood in the field.")
    assert counts(r, "verbatim") == (0, 29, 0, 6)
    assert counts(r, "content") == (0, 28, 0, 6)


def test_case_and_punctuation_count_on_verbatim_only():
    r = error_rates("the house stood in the field", "The house stood in the field.")
    assert counts(r, "verbatim") == (2, 29, 2, 6)
    assert counts(r, "content") == (0, 28, 0, 6)


def test_a_misread_word_counts_on_both_bases():
    r = error_rates("The horse stood in the field.", "The house stood in the field.")
    assert r.verbatim.char_edits == r.content.char_edits == 1
    assert r.verbatim.word_edits == r.content.word_edits == 1


def test_one_wrong_character_is_one_character_and_one_word_on_either_basis():
    r = error_rates("alpha beta gamma", "alpha bita gamma")
    for basis in BASES:
        c = getattr(r, basis)
        assert c.word_edits == 1 and c.char_edits == 1
        assert c.wer > c.cer


def test_a_shared_basis_does_not_make_wer_at_least_cer():
    # a long insertion inside one word: many characters, one word
    r = error_rates("the quick brxxxxxxxxxxxxxxxxxxxxxown fox", "the quick brown fox")
    for basis in BASES:
        c = getattr(r, basis)
        assert c.cer > 1 and c.wer == 0.25


def test_distances_are_levenshtein_without_transpositions():
    r = error_rates("b a", "a b")
    assert r.content.word_edits == 2 and r.verbatim.word_edits == 2
    assert error_rates("ab", "ba").content.char_edits == 2


def test_whitespace_runs_and_line_breaks_are_one_space():
    r = error_rates("The  house\nstood\tin the field.", "The house stood in the field.")
    assert counts(r, "verbatim") == (0, 29, 0, 6)


def test_views():
    assert verbatim_view("  a  b\n") == ("a b", ["a", "b"])
    assert verbatim_view("   ") == ("", [])
    assert content_view('"Quite so," she said -- twice!') == (
        "quite so she said twice", ["quite", "so", "she", "said", "twice"])


# --- content basis: what is and is not disregarded ------------------------

def test_tokens_without_a_letter_or_digit_are_not_words():
    assert content_view("a , . ... - & # ___ *** b")[1] == ["a", "b"]


def test_characters_inside_a_token_are_kept():
    assert content_view("well-known don't e.g. 4,000")[1] == ["well-known", "don't", "e.g", "4,000"]
    r = error_rates("a well known writer", "a well-known writer")
    assert r.content.word_edits == 2 and r.content.char_edits == 1


def test_typographic_apostrophe_is_an_apostrophe_on_content_only():
    r = error_rates("don’t go", "don't go")
    assert r.content.char_edits == 0 and r.content.word_edits == 0
    assert r.verbatim.char_edits == 1 and r.verbatim.word_edits == 1


def test_long_s_is_never_folded():
    r = error_rates("the prieſts", "the priests")
    assert r.content.char_edits == 1 and r.verbatim.char_edits == 1


def test_lowercasing_is_not_casefolding():
    # casefold would equate these; the content basis does not
    assert error_rates("strasse", "straße").content.char_edits == 2


def test_canonically_equivalent_text_is_equal_on_both_bases():
    r = error_rates("café society", "café society")
    assert r.unicode_form == "NFC"
    assert counts(r, "verbatim") == (0, 12, 0, 2) and counts(r, "content") == (0, 12, 0, 2)


def test_a_dropped_accent_is_an_error_wherever_it_stands():
    for pred, ref in [("cafe", "café"), ("cafe", "café"), ("zu", "zuͤ"), ("fur", "fuͤr")]:
        r = error_rates(pred, ref)
        assert r.content.char_edits == 1 and r.content.word_edits == 1, (pred, ref)


def test_underscores_and_symbols_at_token_edges_are_stripped():
    assert content_view("_word_ *word* £5 50% the ___ cat")[1] == ["word", "word", "5", "50", "the", "cat"]


def test_non_ascii_letters_are_letters_at_token_edges():
    assert content_view("«été» Ärger, über.")[1] == ["été", "ärger", "über"]


def test_numbers_of_any_kind_count_as_content():
    # Unicode category N: digits, and also fractions, superscripts, enclosed numbers
    assert content_view("12 ½ ① x²")[1] == ["12", "½", "①", "x²"]
    assert content_view("marshal,⁴")[1] == ["marshal,⁴"]


def test_a_mark_on_trailing_punctuation_goes_with_the_punctuation():
    # struck-through text: U+0336 after every character, the full stop included
    assert content_view("t̶a̶x̶i̶.̶")[1] == ["t̶a̶x̶i̶"]
    assert content_view("a.́")[1] == ["a"]
    assert content_view("great❤️")[1] == ["great"]
    assert error_rates("taxi.̶", "taxi .").content.char_edits == 0


def test_a_leading_combining_mark_is_not_kept():
    assert content_view("́abc")[1] == ["abc"]


def test_exactly_two_typographic_apostrophes_are_read_as_apostrophes():
    assert content_view("donʼt don’t don‘t don`t")[1] == ["don't", "don't", "don‘t", "don`t"]


def test_an_apostrophe_of_any_kind_at_a_token_edge_is_stripped():
    for apostrophe in ("'", "’", "ʼ"):
        assert content_view(f"{apostrophe}tis boys{apostrophe} a {apostrophe} b")[1] == ["tis", "boys", "a", "b"]


def test_dashes_inside_a_token_are_kept_as_they_are():
    assert content_view("yes—no 1914–18")[1] == ["yes—no", "1914–18"]
    assert error_rates("yes—no", "yes-no").content.char_edits == 1


def test_case_is_disregarded_even_where_lowercasing_changes_the_normal_form():
    # T + combining diaeresis lower-cases to t + diaeresis, which NFC composes
    assert error_rates("T̈", "ẗ").content == ErrorCounts(char_edits=0, ref_chars=1, word_edits=0, ref_words=1)
    assert error_rates("H̱ARTUM", "ẖartum").content.char_edits == 0


def test_content_tokens_never_contain_whitespace_or_start_with_punctuation():
    chars, words = content_view("  (a)  'b'  c.\n—d— ")
    assert words == ["a", "b", "c", "d"] and chars == "a b c d"


# --- empty inputs and rates ----------------------------------------------

def test_empty_prediction_deletes_everything():
    r = error_rates("", "The house stood.")
    assert counts(r, "verbatim") == (16, 16, 3, 3)
    assert r.verbatim.cer == 1.0 and r.verbatim.wer == 1.0
    assert counts(error_rates(None, "The house stood."), "verbatim") == (16, 16, 3, 3)


def test_empty_reference_has_no_rate_but_keeps_the_edits():
    r = error_rates("stray text", "")
    assert counts(r, "content") == (10, 0, 2, 0)
    assert r.content.cer is None and r.content.wer is None


def test_none_reference_is_an_empty_reference():
    assert counts(error_rates("a", None), "content") == (1, 0, 1, 0)
    assert counts(error_rates(None, None), "verbatim") == (0, 0, 0, 0)


@pytest.mark.parametrize("value", [0, 0.0, 5, b"abc", ["a"], float("nan")])
def test_only_text_or_none_is_accepted(value):
    with pytest.raises(TypeError, match="must be a string or None"):
        error_rates(value, "abc")
    with pytest.raises(TypeError, match="must be a string or None"):
        error_rates("abc", value)


def test_rates_are_edits_over_reference_length():
    c = ErrorCounts(char_edits=3, ref_chars=12, word_edits=1, ref_words=4)
    assert c.cer == 0.25 and c.wer == 0.25


# --- conventions ----------------------------------------------------------

def test_convention_is_recorded_with_the_counts():
    plain = error_rates("a", "a")
    tok = error_rates("a", "a", TOK)
    assert plain.convention is None
    assert tok.convention == ConventionIdentity(**resolve_convention(TOK).identity)
    assert tok.convention.model_dump() == resolve_convention(TOK).identity
    assert tok.definition == DEFINITION and tok.unicode_form == "NFC"
    inline = error_rates("a", "a", {"name": "mine", "version": "7", "rules": []})
    assert (inline.convention.name, inline.convention.version) == ("mine", "7")


def test_normal_form_comes_before_the_convention():
    # NFD "e + acute" must be composed before the clitic rule looks for a word character
    assert counts(error_rates("café 's", "café's", TOK), "verbatim") == (0, 6, 0, 1)


def test_tokenised_reference_costs_nothing_under_its_convention():
    pred = "It's a pity, he said, that you're late."
    ref = "It 's a pity , he said , that you 're late ."
    without, under = error_rates(pred, ref), error_rates(pred, ref, TOK)
    assert without.content.word_edits == 4 and without.verbatim.word_edits == 10
    assert counts(under, "content") == (0, 36, 0, 8)
    assert counts(under, "verbatim") == (0, 39, 0, 8)


def test_convention_applies_to_both_strings():
    tokenised = "It 's a pity , he said ."
    # both sides become "It's a pity, he said." — 21 characters, 5 words
    assert counts(error_rates(tokenised, tokenised, TOK), "verbatim") == (0, 21, 0, 5)


def test_unknown_convention_raises():
    with pytest.raises(ValueError):
        error_rates("a", "a", "no-such-convention")


# --- pooling --------------------------------------------------------------

def test_pool_sums_counts():
    a = error_rates("the horse stood", "the house stood")
    b = error_rates("", "in the field")
    total = pool([a, b])
    assert counts(total, "content") == (1 + 12, 15 + 12, 1 + 3, 3 + 3)
    assert total.content.wer == pytest.approx(4 / 6)


def test_pool_is_a_micro_average_not_a_mean_of_ratios():
    short = error_rates("x", "a")              # 1 of 1 words wrong
    long = error_rates("a b c d e f g h i", "a b c d e f g h i")  # 0 of 9
    assert pool([short, long]).content.wer == pytest.approx(1 / 10)


def test_pool_sums_every_basis_separately():
    a, b = error_rates("The horse stood.", "the house stood"), error_rates("", "In the field.")
    total = pool([a, b])
    for basis in BASES:
        assert counts(total, basis) == tuple(x + y for x, y in zip(counts(a, basis), counts(b, basis)))
    assert counts(total, "verbatim") != counts(total, "content")


def test_pool_keeps_the_identity_of_what_it_sums():
    r = error_rates("a", "a", TOK)
    total = pool([r, r])
    assert total.convention == r.convention and total.definition == DEFINITION
    assert total.unicode_form == r.unicode_form
    assert pool([error_rates("a", "a")]).convention is None


def test_pool_refuses_every_kind_of_mismatch_and_empty_input():
    plain = error_rates("a", "a")
    x1 = error_rates("a", "a", {"name": "x", "version": "1", "rules": [["b", "c"]]})
    x2 = error_rates("a", "a", {"name": "x", "version": "1", "rules": [["d", "e"]]})   # same name, other rules
    x3 = error_rates("a", "a", {"name": "x", "version": "2", "rules": [["b", "c"]]})   # same rules, other version
    other_definition = ErrorRates(definition="error-rates/0", verbatim=plain.verbatim, content=plain.content)
    other_form = ErrorRates(unicode_form="NFD", verbatim=plain.verbatim, content=plain.content)
    for pair in ([plain, error_rates("a", "a", TOK)], [plain, other_definition], [plain, other_form],
                 [x1, x2], [x1, x3]):
        with pytest.raises(ValueError, match="different"):
            pool(pair)
    with pytest.raises(ValueError, match="at least one"):
        pool([])
    with pytest.raises(TypeError, match="takes ErrorRates"):
        pool([plain, None])


def test_results_are_immutable_and_hashable():
    r = error_rates("a", "a", TOK)
    with pytest.raises(ValidationError):
        r.content.char_edits = 5
    with pytest.raises(ValidationError):
        r.verbatim = r.content
    with pytest.raises(ValidationError):
        r.convention.name = "tampered"
    assert hash(r) == hash(error_rates("a", "a", TOK))


# --- public API -----------------------------------------------------------

def test_public_api():
    assert misnomer.error_rates is error_rates
    assert misnomer.ErrorRates is ErrorRates and misnomer.ErrorCounts is ErrorCounts
    assert "tokenised-english" in misnomer.builtin_conventions()
    assert isinstance(misnomer.builtin_conventions()["tokenised-english"], misnomer.ReferenceConvention)
