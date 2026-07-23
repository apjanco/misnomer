"""Unicode-standard operations for the v1.1 classifier gates.

Two published standards supply all character-level knowledge; no table in
this package is curated by us:

- **Compatibility caseless matching** (Unicode Standard §3.13, D146)
  decides whether two words are the same text modulo display form —
  ligatures, long s, case, numeral presentation forms. Implemented exactly
  as defined: NFKD(casefold(NFKD(casefold(NFD(x))))).
- **UTS #39 confusable skeletons** (vendored ``data/confusables.txt``,
  byte-identical to the file published at unicode.org, version recorded in
  its header) decide whether two words are visually confusable as written.

These answer different questions: compatibility equivalence says what a
character *is* (long s is s); the skeleton says what it *looks like*
(long s resembles f). The classifier consults identity before appearance.

The Python runtime's own Unicode tables (``unicodedata``) may lag the
vendored TR39 file; ``unicode_versions()`` reports both for provenance.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from importlib import resources


def compat_fold(s: str) -> str:
    """Compatibility caseless form per Unicode §3.13 (D146)."""
    x = unicodedata.normalize("NFD", s)
    x = unicodedata.normalize("NFKD", x.casefold())
    x = unicodedata.normalize("NFKD", x.casefold())
    return x


def has_compat_forms(s: str) -> bool:
    """True when the word carries compatibility characters (ligatures,
    long s, presentation forms). Used as the Unicode-derived half of
    archaic-side detection: the side written with display-form characters
    is the side closer to the source typography."""
    return unicodedata.normalize("NFKD", s) != unicodedata.normalize("NFD", s)


@lru_cache(maxsize=1)
def _confusables() -> tuple[dict[str, str], str]:
    """Parse the vendored TR39 confusables table. Returns (map, version)."""
    text = (
        resources.files("misnomer")
        .joinpath("data/confusables.txt")
        .read_text(encoding="utf-8-sig")
    )
    version = "unknown"
    table: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("# Version:"):
            version = line.split(":", 1)[1].strip()
        line = line.split("#")[0].strip()
        if not line:
            continue
        src, tgt, _kind = (part.strip() for part in line.split(";"))
        table[chr(int(src, 16))] = "".join(chr(int(cp, 16)) for cp in tgt.split())
    return table, version


def skeleton(s: str) -> str:
    """TR39 skeleton: NFD, map every character through the confusables
    table, NFD again. Two strings are visually confusable iff their
    skeletons are equal. Applied to the raw glyphs — case- or
    compatibility-folding first would erase exactly the distinctions
    (long s, cross-script letters) whose likeness is being judged."""
    table, _ = _confusables()
    x = unicodedata.normalize("NFD", s)
    x = "".join(table.get(ch, ch) for ch in x)
    return unicodedata.normalize("NFD", x)


def unicode_versions() -> dict[str, str]:
    _, tr39 = _confusables()
    return {
        "unicodedata": unicodedata.unidata_version,
        "tr39_confusables": tr39,
    }
