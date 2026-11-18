"""Reference conventions: declared layout rules, applied to both strings."""

from __future__ import annotations

import dataclasses
import itertools
import json
import time

import pytest

from misnomer.conventions import (
    MAX_PASSES,
    ReferenceConvention,
    builtin_conventions,
    resolve_convention,
)

TOK = "tokenised-english"


@pytest.fixture(scope="module")
def tok() -> ReferenceConvention:
    return resolve_convention(TOK)


# --- resolving ------------------------------------------------------------

def test_none_means_no_convention():
    assert resolve_convention(None) is None


def test_unknown_name_is_an_error_not_a_noop():
    with pytest.raises(ValueError, match="unknown reference convention"):
        resolve_convention("tokenized-english")  # misspelt on purpose


@pytest.mark.parametrize("value", [3, ["a"], ("a", "b"), 1.5])
def test_wrong_type_is_an_error(value):
    with pytest.raises(TypeError):
        resolve_convention(value)


def test_inline_dict_and_instance_resolve():
    inline = resolve_convention({"name": "x", "version": "7", "rules": [[r"\s+,", ","]]})
    assert (inline.name, inline.version) == ("x", "7")
    assert inline.apply("a , b") == "a, b"
    assert resolve_convention(inline) is inline


def test_inline_defaults():
    plain = resolve_convention({"rules": []})
    assert (plain.name, plain.version) == ("custom", "0")
    assert plain.apply("a , b") == "a , b"


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"name": "x"}, "rules"),                                  # no rules
        ({"rules": None}, "list of"),
        ({"rules": "ab"}, "list of"),
        ({"rules": ["ab"]}, "pair of strings"),                    # a string is not a pair
        ({"rules": [" ,", ", "]}, "pair of strings"),              # flat list instead of pairs
        ({"rules": [[r"\s+,", None]]}, "pair of strings"),
        ({"rules": [[1, 2]]}, "pair of strings"),
        ({"rules": [["a", "b", "c"]]}, "pair of strings"),
        ({"rules": [["(", ""]]}, "not a valid"),                   # pattern does not compile
        ({"rules": [["(a)", r"\2"]]}, "not a valid"),              # bad group reference
        ({"rules": [], "descripton": "typo"}, "unknown keys"),
        ({"name": None, "rules": []}, "must be a string"),
        ({"version": 1, "rules": []}, "must be a string"),
    ],
)
def test_malformed_inline_conventions_are_rejected_at_construction(bad, message):
    with pytest.raises(ValueError, match=message):
        resolve_convention(bad)


def test_rules_given_as_lists_are_stored_as_tuples():
    conv = ReferenceConvention("x", "1", "", [["a", "b"]])
    assert conv.rules == (("a", "b"),)
    assert hash(conv) == hash(ReferenceConvention("x", "1", "", (("a", "b"),)))
    assert conv == ReferenceConvention("x", "1", "", (("a", "b"),))


def test_conventions_serialise_without_private_state():
    conv = resolve_convention(TOK)
    as_dict = dataclasses.asdict(conv)
    assert set(as_dict) == {"name", "version", "description", "rules"}
    json.dumps(as_dict)


def test_builtin_registry_is_read_only():
    with pytest.raises(TypeError):
        builtin_conventions()["evil"] = None
    assert sorted(builtin_conventions()) == ["tokenised-english"]


def test_builtins_are_versioned_and_described():
    for conv in builtin_conventions().values():
        assert conv.version and conv.description and conv.rules


# --- identity -------------------------------------------------------------

def test_builtin_identity_is_pinned():
    # Changing a rule changes this hash: bump the convention's version with it.
    assert resolve_convention(TOK).identity == {
        "name": "tokenised-english",
        "version": "1.0",
        "rules_sha256": "704165c9852111cafa95ac35ada5c660a3eb43259782e5db1a8a090871e0da4e",
    }


def test_identity_distinguishes_rule_sets_order_and_replacements():
    digest = lambda *rules: ReferenceConvention("x", "1", "", tuple(rules)).rules_sha256
    assert digest(("a", "b")) != digest(("c", "b"))
    assert digest(("a", "b")) != digest(("a", "c"))
    assert digest(("a", "b"), ("c", "d")) != digest(("c", "d"), ("a", "b"))
    assert len(digest(("a", "b"))) == 64
    conv = ReferenceConvention("n", "9", "", (("a", "b"),))
    assert conv.identity == {"name": "n", "version": "9", "rules_sha256": conv.rules_sha256}


# --- applying -------------------------------------------------------------

def test_rules_are_compiled_without_flags():
    assert resolve_convention({"rules": [["a", "b"]]}).apply("A") == "A"             # no IGNORECASE
    assert resolve_convention({"rules": [[r"[ ]+$", ""]]}).apply("a \nb") == "a \nb"  # no MULTILINE
    assert resolve_convention({"rules": [[r"\w \w", "x"]]}).apply("é é") == "x"       # no ASCII


def test_rules_apply_in_order():
    conv = resolve_convention({"rules": [["a", "b"], ["b", "c"]]})
    assert conv.apply("a") == "c"


def test_rules_are_applied_until_nothing_changes():
    conv = resolve_convention({"rules": [["aa", "a"]]})
    assert conv.apply("aaaaaaaa") == "a"


def test_rules_that_never_settle_are_an_error():
    with pytest.raises(ValueError, match="did not settle"):
        ReferenceConvention("loop", "1", "", (("a", "aa"),)).apply("a")
    assert MAX_PASSES >= 3


@pytest.mark.parametrize(
    "tokenised, plain",
    [
        ("It 's a pity , he said , that you 're late .", "It's a pity, he said, that you're late."),
        ("I 'm sure he 'll come ; they 'd say we 've won !", "I'm sure he'll come; they'd say we've won!"),
        ("He did n't go ( at least not then ) .", "He didn't go (at least not then)."),
        ("wait ... what ?", "wait... what?"),
        ("the M.P. 's view", "the M.P.'s view"),
        ("it ’s a change", "it’s a change"),
        ("[ b ] { c }", "[b] {c}"),
        ("a . )", "a.)"),
        ("IT 'S HERE", "IT'S HERE"),
        ("DID N'T", "DIDN'T"),
        ("did n’t", "didn’t"),
        ("s . 'S", "s.'S"),
        ("café 's", "café's"),
        ("a\nb\t  c", "a b c"),
    ],
)
def test_tokenised_layout_is_undone(tok, tokenised, plain):
    assert tok.apply(tokenised) == plain
    assert tok.apply(plain) == plain  # ordinary spacing is left alone


@pytest.mark.parametrize(
    "tokenised, plain",
    [
        ('He said " No " twice .', 'He said "No" twice.'),
        ('( he cried " stop " ! )', '(he cried "stop"!)'),
        ('( " yes " . )', '("yes".)'),
        ('He asked ( did she say " no " ? ) twice .', 'He asked (did she say "no"?) twice.'),
        ('" Quite so , " she said .', '"Quite so," she said.'),
    ],
)
def test_both_layouts_of_a_quotation_converge(tok, tokenised, plain):
    assert tok.apply(tokenised).split() == tok.apply(plain).split()


def test_double_quotes_are_detached_typographic_ones_too(tok):
    assert tok.apply('He said "No" twice.').split() == ["He", "said", '"', "No", '"', "twice."]
    assert tok.apply("He said “No” twice.").split() == ["He", "said", "“", "No", "”", "twice."]


def test_quote_inside_a_word_is_left_alone(tok):
    assert tok.apply('the town of Ko"then , near') == 'the town of Ko"then, near'


@pytest.mark.parametrize(
    "text",
    [
        "dragged on ,with",            # punctuation glued to the next word is not a layout token
        "return... ...which shows",
        "wake .7 grams",
        "perme( Care",
        "any ?dabs?",
        "a )b",
        "bearing ),and this",
        "a ).b",
        "word ]:next",
        "x.( y",
        "a .)a",
        "he replied 'so what' again",  # single quotation marks are not clitics
        "tell 'em now",
        "the '30s were hard",
        "a 'd' sound",
        ". 's",
        ", n't",
        "a n'tx",
        "can 't",
        "wait … what",
        "he said ’ so",
    ],
)
def test_rules_do_not_merge_or_split_words(tok, text):
    assert tok.apply(text).split() == text.split()


IDEMPOTENCE_CASES = [
    "It 's a pity , he said , that you 're late .",
    'He said " No " twice .',
    'word ." next',
    '". )',
    '"; )',
    '( " yes " . )',
    '("yes".)',
    "( a ) [ b ] { c }",
    "x ... ... y , ; : ! ?",
    "",
    "   ",
]


@pytest.mark.parametrize("text", IDEMPOTENCE_CASES)
def test_idempotent(tok, text):
    once = tok.apply(text)
    assert tok.apply(once) == once


def test_idempotent_and_whitespace_only_exhaustively(tok):
    # every string of up to five atoms over the characters the rules react to
    atoms = ['"', " ", "a", ".", ")", "(", "'s", ",", "n't", "!"]
    squeeze = lambda s: "".join(s.split())
    for length in range(1, 6):
        for combo in itertools.product(atoms, repeat=length):
            text = "".join(combo)
            once = tok.apply(text)
            assert tok.apply(once) == once, text
            assert squeeze(once) == squeeze(text), text


def test_only_whitespace_is_ever_changed(tok):
    text = "It 's \" odd \" , is n't it ? ( yes ) [ no ] … 'tis so . “ q ”"
    squeeze = lambda s: "".join(s.split())
    assert squeeze(tok.apply(text)) == squeeze(text)


def test_long_whitespace_runs_are_handled_in_linear_time(tok):
    started = time.perf_counter()
    assert tok.apply(" " * 50_000 + "a , b" + "\n" * 50_000) == " a, b "
    tok.apply(" ." * 20_000)
    tok.apply('"' * 20_000)
    assert time.perf_counter() - started < 5.0  # quadratic behaviour took minutes
