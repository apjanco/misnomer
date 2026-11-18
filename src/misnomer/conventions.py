"""Reference conventions — declared statements of how a reference text is
laid out.

A reference transcription is not only a sequence of words; it also follows
a layout convention. Tagged corpora, for instance, store punctuation marks
and clitics as separate tokens ("It 's a pity , he said ."). A system that
writes the same words with ordinary spacing has misread nothing, yet a
character or word comparison against such a reference charges it for every
space. Like an orthographic tradition (see ``traditions.py``), a layout
convention is a fact about how the reference was prepared, not about the
characters in it, so it must be declared rather than guessed: a convention
is named, versioned data, selected explicitly by the caller and recorded
with every figure computed under it. Nothing is applied by default.

A convention is an ordered list of ``(pattern, replacement)`` regular-
expression rules. The whole list is applied to BOTH strings of a comparison,
repeatedly, until the text no longer changes — so applying a convention to
its own output never changes it again. Patterns are compiled without flags;
a rule that needs one states it inline (``(?i)``). Built-in conventions live
in ``data/conventions.json``; a custom convention is passed inline as a dict
with the keys ``name``, ``version``, ``description`` and ``rules``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Mapping

# A convention whose rules have not settled after this many passes is
# reported as an error instead of being applied half-way.
MAX_PASSES = 8

_INLINE_KEYS = {"name", "version", "description", "rules"}


@dataclass(frozen=True)
class ReferenceConvention:
    name: str
    version: str
    description: str
    # Ordered (regex pattern, replacement) pairs, applied with re.sub.
    rules: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for label, value in (("name", self.name), ("version", self.version),
                             ("description", self.description)):
            if not isinstance(value, str):
                raise ValueError(f"reference convention: {label} must be a string, got {value!r}")
        if isinstance(self.rules, (str, bytes)) or not isinstance(self.rules, (list, tuple)):
            raise ValueError(
                f"convention {self.name!r}: rules must be a list of (pattern, replacement) pairs"
            )
        normalised = []
        compiled = []
        for rule in self.rules:
            if (
                isinstance(rule, (str, bytes))
                or not isinstance(rule, (list, tuple))
                or len(rule) != 2
                or not all(isinstance(part, str) for part in rule)
            ):
                raise ValueError(
                    f"convention {self.name!r}: each rule must be a (pattern, replacement) "
                    f"pair of strings, got {rule!r}"
                )
            pattern, replacement = rule
            try:
                regex = re.compile(pattern)
                regex.sub(replacement, "")  # a bad group reference fails here, not in use
            except (re.error, IndexError) as exc:
                raise ValueError(
                    f"convention {self.name!r}: rule {rule!r} is not a valid "
                    f"pattern/replacement pair: {exc}"
                ) from exc
            normalised.append((pattern, replacement))
            compiled.append((regex, replacement))
        # Stored as tuples so the convention is hashable; the compiled patterns
        # are a private cache, not a field (kept out of asdict/eq/hash/repr).
        object.__setattr__(self, "rules", tuple(normalised))
        object.__setattr__(self, "_compiled", tuple(compiled))

    def apply(self, text: str) -> str:
        """Apply the rules, in order, until *text* no longer changes."""
        current = text
        for _ in range(MAX_PASSES):
            out = current
            for regex, replacement in self._compiled:
                out = regex.sub(replacement, out)
            if out == current:
                return out
            current = out
        raise ValueError(
            f"convention {self.name!r} did not settle within {MAX_PASSES} passes; "
            "its rules keep rewriting their own output"
        )

    @property
    def rules_sha256(self) -> str:
        """Hash of the ordered rule list: two conventions with the same name
        and version but different rules are distinguishable in any output."""
        payload = json.dumps([list(rule) for rule in self.rules], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def identity(self) -> dict[str, str]:
        """What is recorded alongside figures computed under this convention."""
        return {"name": self.name, "version": self.version, "rules_sha256": self.rules_sha256}


@lru_cache(maxsize=1)
def _builtin() -> Mapping[str, ReferenceConvention]:
    raw = json.loads(
        resources.files("misnomer")
        .joinpath("data/conventions.json")
        .read_text(encoding="utf-8")
    )
    return MappingProxyType({
        name: ReferenceConvention(
            name=name,
            version=spec["version"],
            description=spec["description"],
            rules=tuple(tuple(rule) for rule in spec["rules"]),
        )
        for name, spec in raw.items()
    })


def builtin_conventions() -> Mapping[str, ReferenceConvention]:
    """The built-in conventions, by name (read-only)."""
    return _builtin()


def resolve_convention(
    convention: "str | dict | ReferenceConvention | None",
) -> ReferenceConvention | None:
    """Resolve a convention given by built-in name, inline dict, or instance.

    ``None`` means no convention: the strings are compared as supplied. An
    unknown name, an unknown key or a malformed rule is an error, never a
    silent no-op.
    """
    if convention is None:
        return None
    if isinstance(convention, ReferenceConvention):
        return convention
    if isinstance(convention, str):
        conventions = builtin_conventions()
        if convention not in conventions:
            raise ValueError(
                f"unknown reference convention {convention!r}; built-ins: {sorted(conventions)}"
            )
        return conventions[convention]
    if isinstance(convention, dict):
        unknown = set(convention) - _INLINE_KEYS
        if unknown:
            raise ValueError(
                f"inline reference convention has unknown keys {sorted(unknown)}; "
                f"allowed: {sorted(_INLINE_KEYS)}"
            )
        if "rules" not in convention:
            raise ValueError("an inline reference convention needs a 'rules' list")
        return ReferenceConvention(
            name=convention.get("name", "custom"),
            version=convention.get("version", "0"),
            description=convention.get("description", "inline custom convention"),
            rules=convention["rules"],
        )
    raise TypeError(
        "reference convention must be a built-in name, a dict, a ReferenceConvention "
        f"or None, got {type(convention).__name__}"
    )
