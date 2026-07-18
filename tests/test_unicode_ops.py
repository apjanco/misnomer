"""Unicode-standard operations: compatibility caseless matching (§3.13)
and TR39 confusable skeletons. These tests lock the *standards'* judgments
so a future data-file update that changes a load-bearing fact fails loudly
instead of silently reclassifying."""

from __future__ import annotations

from misnomer.unicode_ops import (
    _confusables,
    compat_fold,
    has_compat_forms,
    skeleton,
    unicode_versions,
)


def test_compat_fold_derives_display_forms():
    # Long s, sharp s, ligatures, numeral presentation forms: all folded by
    # the Unicode Standard's own operation, no table of ours involved.
    assert compat_fold("ſ") == "s"
    assert compat_fold("ß") == "ss"
    assert compat_fold("ﬁne") == "fine"
    assert compat_fold("Ⅳ") == "iv"
    assert compat_fold("The") == "the"


def test_compat_fold_preserves_meaningful_marks():
    # Diacritics are not display forms; they never fold in the base layer.
    assert compat_fold("café") != "cafe"
    # æ is a letter to Unicode — its digraph status is tradition-dependent
    # and therefore belongs to a declared profile, not to this layer.
    assert compat_fold("æther") != "aether"


def test_has_compat_forms():
    assert has_compat_forms("prieſts")
    assert has_compat_forms("ﬁne")
    assert not has_compat_forms("priests")
    assert not has_compat_forms("café")
    # ß has no compatibility decomposition (casefold handles it instead),
    # so it does not mark a word as archaic-side.
    assert not has_compat_forms("straße")


def test_confusables_table_is_the_full_standard():
    table, version = _confusables()
    assert version == "16.0.0"
    assert len(table) > 6000  # the vendored file, not a curated subset


def test_skeleton_locks_tr39_judgments():
    table, _ = _confusables()
    # Visual likeness is distinct from identity: long s LOOKS like f...
    assert table["ſ"] == "f"
    # ...the minim class is in the standard (m is confusable with rn)...
    assert table["m"] == "rn"
    # ...Cyrillic a is identical-by-design to Latin a...
    assert table["а"] == "a"
    # ...but TR39 grades lookalikes: Cyrillic ka maps to kra, NOT Latin k —
    # the standard judges к/k distinguishable. We accept that judgment.
    assert table["к"] == "ĸ" and "k" not in table
    # And no Unicode source relates u/v or i/j: positional usage is
    # orthographic tradition, not character knowledge. Load-bearing for the
    # default-modern design — if a data update ever changes this, fail loud.
    assert "v" not in table and "u" not in table and "j" not in table


def test_skeleton_word_level():
    assert skeleton("mаp") == skeleton("map")  # Cyrillic а
    assert skeleton("bam") == skeleton("barn")  # minim class
    assert skeleton("prieſts") == skeleton("priefts")  # long-s/f likeness
    assert skeleton("Luкe") != skeleton("Luke")  # TR39: к not a k lookalike
    assert skeleton("gaue") != skeleton("gave")  # u/v is not a visual matter


def test_unicode_versions_reports_both_authorities():
    versions = unicode_versions()
    assert set(versions) == {"unicodedata", "tr39_confusables"}
    assert versions["tr39_confusables"] == "16.0.0"
