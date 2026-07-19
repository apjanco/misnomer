"""Canonical analysis layer over rescored IAM outputs.

EVERY statistic and EVERY example dump must go through this module — the
filters here are the single source of truth, so numbers and exhibits can
never silently diverge (the process failure this module exists to prevent).

Row-level filters (line excluded from word-level statistics):
  refusal/empty         -> counted as coverage failure
  degenerate output     -> pred_len > 2*gt_len + 20 (CoT leaks, loops)
  hallucinated line     -> document_error_type == "hallucinated"
Substitution-level filters:
  SEGMENTATION class    -> structural (character anchoring, scorer-level)
  GT-suspect            -> same (index, gt, pred) from >= GT_SUSPECT_MIN
                           systems; flagged for image adjudication
Sieve classes (invisible substitutions): SEMANTIC, OBVIOUS.
Visible classes (a text-side procedure can shortlist them): GARBLE,
HOMOGLYPH, NUMERIC — workflow-relative; see trap_rate(visible=...).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).parent

SIEVE = {"SEMANTIC", "OBVIOUS"}
VISIBLE_DEFAULT = frozenset({"GARBLE", "HOMOGLYPH", "NUMERIC", "SEGMENTATION"})
GT_SUSPECT_MIN = 5

# system -> (results file stem, default rescore dir, restrict-to-test-split)
SYSTEMS: dict[str, tuple[str, bool]] = {
    "Qwen3-VL-8B-Instruct": ("Qwen3-VL-8B-Instruct", False),
    "chandra": ("chandra", False),
    "chatgpt-gpt4o": ("chatgpt-gpt4o", False),
    "chatgpt5": ("chatgpt5", False),
    "claude-vision": ("claude-vision", False),
    "gemini-portkey": ("gemini-portkey", False),
    "tesseract": ("tesseract", False),
    "kraken-mccatmus": ("kraken-mccatmus", False),
    "pylaia-iam": ("pylaia-iam", True),
    "pylaia-iam-lm": ("pylaia-iam-lm", True),
}
VLM_SYSTEMS = {"Qwen3-VL-8B-Instruct", "chandra", "chatgpt-gpt4o",
               "chatgpt5", "claude-vision", "gemini-portkey"}


@dataclass
class Line:
    index: int
    gt_words: int
    excluded: str | None  # None = included; else reason
    visible_subs: int
    substitutions: list[dict] = field(default_factory=list)  # non-excluded subs


@dataclass
class SystemData:
    name: str
    lines: list[Line]
    n_scored: int
    n_refusal: int
    n_degenerate: int
    n_hallucinated: int
    gt_suspect_hits: int

    @property
    def coverage(self) -> float:
        return 1 - self.n_refusal / max(1, self.n_scored)


def _raw_meta(raw: str):
    gtw, split = {}, {}
    for line in open(EVAL_DIR / f"{raw}_results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        i = int(r.get("index", r.get("image_idx", r.get("image_path"))))
        gtw[i] = len((r.get("ground_truth") or "").split())
        split[i] = r.get("split")
    return gtw, split


def _collect_convergence(scored_dir: Path) -> set[tuple[int, str, str]]:
    conv: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for name, (raw, test_only) in SYSTEMS.items():
        path = scored_dir / f"{raw}_rescored.jsonl"
        if not path.exists():
            continue
        _, split = _raw_meta(raw)
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if test_only and split.get(rec["index"]) != "test":
                continue
            for s in rec["substitutions"]:
                conv[(rec["index"], s["ground_truth_word"].casefold(),
                      s["predicted_word"].casefold())].add(name)
    return {k for k, v in conv.items() if len(v) >= GT_SUSPECT_MIN}


def load_system(name: str, scored_dir: str | Path,
                gt_suspect: set | None = None) -> SystemData:
    raw, test_only = SYSTEMS[name]
    scored_dir = Path(scored_dir)
    if gt_suspect is None:
        gt_suspect = _collect_convergence(scored_dir)
    gtw, split = _raw_meta(raw)

    lines: list[Line] = []
    n_scored = n_ref = n_deg = n_hal = n_sus = 0
    for line in open(scored_dir / f"{raw}_rescored.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        i = rec["index"]
        if test_only and split.get(i) != "test":
            continue
        n_scored += 1
        if rec["is_refusal"]:
            n_ref += 1
            lines.append(Line(i, gtw[i], "refusal", 0))
            continue
        if "pred_len" in rec and rec["pred_len"] > 2 * rec["gt_len"] + 20:
            n_deg += 1
            lines.append(Line(i, gtw[i], "degenerate", 0))
            continue
        if rec.get("document_error_type") == "hallucinated":
            n_hal += 1
            lines.append(Line(i, gtw[i], "hallucinated", 0))
            continue
        subs = []
        visible = 0
        for s in rec["substitutions"]:
            if (i, s["ground_truth_word"].casefold(),
                    s["predicted_word"].casefold()) in gt_suspect:
                n_sus += 1
                continue
            if s["error_type"] in VISIBLE_DEFAULT:
                visible += 1
            subs.append(s)
        lines.append(Line(i, gtw[i], None, visible, subs))
    return SystemData(name, lines, n_scored, n_ref, n_deg, n_hal, n_sus)


# ---------------------------------------------------------------------------
# metrics (all computed over included lines only)
# ---------------------------------------------------------------------------

def stratum_of(sub: dict, position_hint: int | None = None) -> str:
    g = sub["ground_truth_word"]
    if any(ch.isnumeric() for ch in g):
        return "number"
    pos = sub.get("context_words")
    if g[:1].isupper() and (pos is None or pos > 0):
        return "name"
    return "common"


def _totals(data: SystemData, deadband: float = 0.0,
            stratum: str | None = None):
    words = npass = nplus = nsubs = 0
    for ln in data.lines:
        if ln.excluded:
            continue
        words += ln.gt_words
        for s in ln.substitutions:
            if stratum and stratum_of(s) != stratum:
                continue
            nsubs += 1
            if s["error_type"] in SIEVE:
                npass += 1
                if s["delta"] is not None and s["delta"] > deadband:
                    nplus += 1
    return words, npass, nplus, nsubs


def ier(data: SystemData, **kw) -> float:
    words, npass, _, _ = _totals(data, **kw)
    return 1000 * npass / max(1, words)


def ier_plus(data: SystemData, deadband: float = 0.0, **kw) -> float:
    words, _, nplus, _ = _totals(data, deadband=deadband, **kw)
    return 1000 * nplus / max(1, words)


def smooth_share(data: SystemData, deadband: float = 0.0, **kw) -> float:
    _, npass, nplus, _ = _totals(data, deadband=deadband, **kw)
    return nplus / npass if npass else 0.0


def smooth_per_100_subs(data: SystemData, deadband: float = 0.0, **kw) -> float:
    _, _, nplus, nsubs = _totals(data, deadband=deadband, **kw)
    return 100 * nplus / nsubs if nsubs else 0.0


def trap_rate(data: SystemData) -> tuple[float, int]:
    """(share of clean-looking lines containing >=1 invisible substitution,
    number of clean-looking lines)."""
    clean = trapped = 0
    for ln in data.lines:
        if ln.excluded or ln.visible_subs:
            continue
        clean += 1
        if any(s["error_type"] in SIEVE for s in ln.substitutions):
            trapped += 1
    return (trapped / clean if clean else float("nan")), clean


def bootstrap_ci(data: SystemData, metric, n_boot: int = 1000,
                 seed: int = 20260718, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap over lines (the sampling unit)."""
    rng = random.Random(seed)
    included = [ln for ln in data.lines if not ln.excluded]
    vals = []
    for _ in range(n_boot):
        sample = [included[rng.randrange(len(included))] for _ in included]
        proxy = SystemData(data.name, sample, len(sample), 0, 0, 0, 0)
        vals.append(metric(proxy))
    vals.sort()
    lo = vals[int((1 - level) / 2 * n_boot)]
    hi = vals[int((1 + level) / 2 * n_boot) - 1]
    return lo, hi


def summary_table(scored_dirs: dict[str, str | Path]) -> str:
    gt_suspect: set = set()
    dirs = {Path(d) for d in scored_dirs.values()}
    for d in dirs:
        gt_suspect |= _collect_convergence(d)
    out = [f"{'system':<22} {'cov':>5} {'IER':>7} {'IER+':>6} {'share':>6} "
           f"{'sm/100':>7} {'trap':>12} {'seg':>5}"]
    for name, sd in scored_dirs.items():
        data = load_system(name, sd, gt_suspect)
        segs = sum(1 for ln in data.lines if not ln.excluded
                   for s in ln.substitutions if s["error_type"] == "SEGMENTATION")
        t, nclean = trap_rate(data)
        out.append(
            f"{name:<22} {data.coverage:>5.0%} {ier(data):>7.1f} "
            f"{ier_plus(data):>6.1f} {smooth_share(data):>6.0%} "
            f"{smooth_per_100_subs(data):>7.1f} "
            f"{t:>6.0%} (n={nclean:<4}) {segs:>4}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "evaluation/derived/full_v11"
    print(summary_table({name: base for name in SYSTEMS}))


# ---------------------------------------------------------------------------
# Smooth-class precision screens (added after the 2026-07-19 manual sweep of
# ~330 stratified smooth substitutions across all ten systems). Applied to
# IER+ as a sensitivity bound; the sweep's precision estimates: VLM smooth
# classes ~76-86% clean, conventional systems ~35-50% (their smooth items
# are mostly positional pairings inside wrecked lines).
# ---------------------------------------------------------------------------

def smooth_screens(sub: dict, gt_tokens: list[str], pred_line: str) -> str | None:
    """Return the artifact class of a Δ>0 sieve substitution, or None if clean.

    "fragment-edge": the GT word is the line's first/last token and one word
        is an affix of the other — IAM ground truth keeps hyphen-broken
        line-edge fragments ("panionship"), which models complete; the
        completion earns a large spurious Δ.
    "diaeresis": IAM encodes umlaut/diaeresis as a double quote inside the
        word (nai"vety, Ko"then); modern spellings are convention, not error.
    "boiler-residue": reasoning/markup markers in the prediction line that
        survived the length filter (backticks, bullet arrows).
    "low-anchor": anchor_confidence < 0.25 — the pairing is positional, not
        supported by ink.
    """
    g = sub["ground_truth_word"]
    stripped = [t.strip(".,\"'") for t in gt_tokens]
    p, gl = sub["predicted_word"].casefold(), g.casefold()
    if stripped and (g == stripped[0] or g == stripped[-1]) and len(g) >= 2 \
            and p != gl and (p.endswith(gl) or p.startswith(gl)
                             or gl.endswith(p) or gl.startswith(p)):
        return "fragment-edge"
    if '"' in g.strip('"'):
        return "diaeresis"
    if any(m in pred_line for m in ("```", "`", "* ", "->")):
        return "boiler-residue"
    conf = sub.get("anchor_confidence")
    if conf is not None and conf < 0.25:
        return "low-anchor"
    return None
