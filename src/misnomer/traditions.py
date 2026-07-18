"""Orthographic tradition profiles — machine-readable statements of
transcription principles.

A profile declares the equivalences that no character database can supply
(u/v positional usage, the digraph status of æ) because they are facts
about scribal and printing practice, not about characters: Unicode
deliberately refuses to decompose æ precisely because its status differs
by language. Every critical edition states such principles in its front
matter; here they are data — named, versioned, and selectable — so the
classifier can never apply a convention silently.

Built-in profiles live in ``data/traditions.json``. Custom profiles come
in via ``ScorerConfig.tradition`` (a name or an inline dict) or, for
backward compatibility, ``ScorerConfig.allograph_map``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from misnomer.unicode_ops import compat_fold, has_compat_forms

DEFAULT_TRADITION = "modern"


@dataclass(frozen=True)
class TraditionProfile:
    name: str
    version: str
    description: str
    # Ordered literal replacements applied to the compatibility-folded word.
    equivalences: tuple[tuple[str, str], ...] = ()
    # Characters/sequences whose presence marks a word as the archaic side
    # (used by custom allograph_map profiles; built-ins rely on Unicode
    # compatibility forms plus dictionary membership instead).
    markers: frozenset[str] = field(default_factory=frozenset)

    def fold(self, word: str) -> str:
        out = compat_fold(word)
        for old, new in self.equivalences:
            out = out.replace(old, new)
        return out

    def marks_archaic(self, word: str) -> bool:
        return has_compat_forms(word) or any(m in word for m in self.markers)


@lru_cache(maxsize=1)
def builtin_traditions() -> dict[str, TraditionProfile]:
    raw = json.loads(
        resources.files("misnomer")
        .joinpath("data/traditions.json")
        .read_text(encoding="utf-8")
    )
    return {
        name: TraditionProfile(
            name=name,
            version=spec["version"],
            description=spec["description"],
            equivalences=tuple((a, b) for a, b in spec["equivalences"]),
        )
        for name, spec in raw.items()
    }


def resolve_tradition(cfg) -> TraditionProfile:
    """Resolve ScorerConfig into a TraditionProfile.

    Precedence: cfg.tradition (name | inline dict | profile instance),
    then cfg.allograph_map (legacy custom map), then the "modern" default —
    i.e. no period assumptions unless the caller declares them.
    """
    tradition = getattr(cfg, "tradition", None)
    if isinstance(tradition, TraditionProfile):
        return tradition
    if isinstance(tradition, str):
        profiles = builtin_traditions()
        if tradition not in profiles:
            raise ValueError(
                f"unknown tradition {tradition!r}; built-ins: {sorted(profiles)}"
            )
        return profiles[tradition]
    if isinstance(tradition, dict):
        return TraditionProfile(
            name=str(tradition.get("name", "custom")),
            version=str(tradition.get("version", "0")),
            description=str(tradition.get("description", "inline custom profile")),
            equivalences=tuple(
                (str(a), str(b)) for a, b in tradition.get("equivalences", ())
            ),
            markers=frozenset(tradition.get("markers", ())),
        )
    allograph_map = getattr(cfg, "allograph_map", None)
    if allograph_map:
        return TraditionProfile(
            name="custom-allograph-map",
            version="0",
            description="derived from ScorerConfig.allograph_map (legacy)",
            equivalences=tuple(
                (k.casefold(), v.casefold()) for k, v in allograph_map.items()
            ),
            markers=frozenset(allograph_map.keys()),
        )
    return builtin_traditions()[DEFAULT_TRADITION]
