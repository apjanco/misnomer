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
    # raw texts, carried so the smooth screens can run through this module
    gt_tokens: list[str] = field(default_factory=list)
    pred_line: str = ""
    insertion_ratio: float = 0.0


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
    """index -> (gt token list, predicted text, split)."""
    meta = {}
    for line in open(EVAL_DIR / f"{raw}_results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        i = int(r.get("index", r.get("image_idx", r.get("image_path"))))
        meta[i] = ((r.get("ground_truth") or "").split(),
                   r.get("predicted_text") or "", r.get("split"))
    return meta


def _collect_convergence(scored_dir: Path) -> set[tuple[int, str, str]]:
    conv: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for name, (raw, test_only) in SYSTEMS.items():
        path = scored_dir / f"{raw}_rescored.jsonl"
        if not path.exists():
            continue
        meta = _raw_meta(raw)
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if test_only and meta[rec["index"]][2] != "test":
                continue
            for s in rec["substitutions"]:
                conv[(rec["index"], s["ground_truth_word"].casefold(),
                      s["predicted_word"].casefold())].add(name)
    return {k for k, v in conv.items() if len(v) >= GT_SUSPECT_MIN}


def load_system(name: str, scored_dir: str | Path,
                gt_suspect: set | None = None,
                drop_hallucinated: bool = False) -> SystemData:
    """Load one system's rescored lines.

    drop_hallucinated (default False since 2026-08-06): the document-level
    "hallucinated" label is a similarity test built for a VLM failure mode
    — a fluent invented passage, where word alignment carries no meaning.
    Applied uniformly it misfires on conventional-engine wreckage, which
    IS positionally alignable and which the GARBLE class exists to count:
    it removed 62% of tesseract's lines and 61% of kraken's against 0.1-2%
    for the VLMs, so those two systems were scored on their best third
    while the VLMs were scored on everything. Over-long output (the case
    that genuinely breaks alignment, including reasoning leaks) is still
    excluded by the degenerate-length rule above. Pass True to reproduce
    the pre-2026-08-06 behaviour.
    """
    raw, test_only = SYSTEMS[name]
    scored_dir = Path(scored_dir)
    if gt_suspect is None:
        gt_suspect = _collect_convergence(scored_dir)
    meta = _raw_meta(raw)

    lines: list[Line] = []
    n_scored = n_ref = n_deg = n_hal = n_sus = 0
    for line in open(scored_dir / f"{raw}_rescored.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        i = rec["index"]
        gt_tokens, pred_line, split = meta[i]
        if test_only and split != "test":
            continue
        n_scored += 1
        n_words = len(gt_tokens)
        if rec["is_refusal"]:
            n_ref += 1
            lines.append(Line(i, n_words, "refusal", 0))
            continue
        if "pred_len" in rec and rec["pred_len"] > 2 * rec["gt_len"] + 20:
            n_deg += 1
            lines.append(Line(i, n_words, "degenerate", 0))
            continue
        if rec.get("document_error_type") == "hallucinated":
            n_hal += 1
            if drop_hallucinated:
                lines.append(Line(i, n_words, "hallucinated", 0))
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
        lines.append(Line(i, n_words, None, visible, subs, gt_tokens, pred_line,
                          rec.get("insertion_ratio", 0.0)))
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
            stratum: str | None = None, screened: bool = False):
    """Counting core. Decisions encoded here:

    - SEGMENTATION events are never substitutions: excluded from the
      substitution denominator entirely (they are alignment facts, not
      readings; counting them deflated conventional systems' per-error
      rates — i.e. biased *against* the paper's claim — but was wrong).
    - screened=True additionally drops sieve-class substitutions whose
      smooth_screens() verdict is non-None (fragment-edge, diaeresis,
      boiler-residue, low-anchor) from ALL counts. Report both modes.
    """
    words = npass = nplus = nsubs = 0
    for ln in data.lines:
        if ln.excluded:
            continue
        words += ln.gt_words
        for s in ln.substitutions:
            if s["error_type"] == "SEGMENTATION":
                continue
            if stratum and stratum_of(s) != stratum:
                continue
            if screened and s["error_type"] in SIEVE and \
                    smooth_screens(s, ln.gt_tokens, ln.pred_line) is not None:
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
    number of clean-looking lines).

    Clean-looking = no visible substitution, no insertions, and predicted
    word count within 1 of the reference's alphanumeric token count
    (2026-07-24; supersedes the substitution-only definition). Without the
    insertion/length guard, truncated stubs and insertion-borne garble
    counted as clean — for conventional engines most of the "clean" set
    was such lines (tesseract 129 -> 58, kraken 63 -> 58). VLM rates are
    unchanged by the guard; their n shrinks by ~3%."""
    clean = trapped = 0
    for ln in data.lines:
        if ln.excluded or ln.visible_subs:
            continue
        gt_ws = sum(1 for t in ln.gt_tokens if any(c.isalnum() for c in t))
        if ln.insertion_ratio != 0 or abs(len(ln.pred_line.split()) - gt_ws) > 1:
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


def summary_table(scored_dirs: dict[str, str | Path], ci: bool = False) -> str:
    """Main table, raw and screened side by side. ci=True adds a 95% line-
    bootstrap interval on the screened IER+ (slower)."""
    gt_suspect: set = set()
    dirs = {Path(d) for d in scored_dirs.values()}
    for d in dirs:
        gt_suspect |= _collect_convergence(d)
    hdr = (f"{'system':<22} {'cov':>5} {'IER':>7} {'IER+':>6} {'IER+scr':>8} "
           f"{'share':>6} {'sh-scr':>7} {'sm/100scr':>10} {'trap':>12} {'seg':>5}")
    out = [hdr]
    for name, sd in scored_dirs.items():
        data = load_system(name, sd, gt_suspect)
        segs = sum(1 for ln in data.lines if not ln.excluded
                   for s in ln.substitutions if s["error_type"] == "SEGMENTATION")
        t, nclean = trap_rate(data)
        row = (
            f"{name:<22} {data.coverage:>5.0%} {ier(data):>7.1f} "
            f"{ier_plus(data):>6.1f} {ier_plus(data, screened=True):>8.2f} "
            f"{smooth_share(data):>6.0%} {smooth_share(data, screened=True):>7.0%} "
            f"{smooth_per_100_subs(data, screened=True):>10.1f} "
            f"{t:>6.0%} (n={nclean:<4}) {segs:>4}")
        if ci:
            lo, hi = bootstrap_ci(data, lambda d: ier_plus(d, screened=True))
            row += f"  CI[{lo:.2f},{hi:.2f}]"
        out.append(row)
    return "\n".join(out)


def sensitivity_tables(scored_dirs: dict[str, str | Path]) -> str:
    """Deadband sensitivity (Δ>0 / >0.5 / >1.0) and entity strata, screened."""
    gt_suspect: set = set()
    for d in {Path(d) for d in scored_dirs.values()}:
        gt_suspect |= _collect_convergence(d)
    out = [f"{'system':<22} {'IER+ Δ>0':>9} {'Δ>0.5':>7} {'Δ>1.0':>7} "
           f"{'names':>7} {'numbers':>8} {'common':>7}"]
    for name, sd in scored_dirs.items():
        data = load_system(name, sd, gt_suspect)
        out.append(
            f"{name:<22} {ier_plus(data, screened=True):>9.2f} "
            f"{ier_plus(data, deadband=0.5, screened=True):>7.2f} "
            f"{ier_plus(data, deadband=1.0, screened=True):>7.2f} "
            f"{ier_plus(data, screened=True, stratum='name'):>7.2f} "
            f"{ier_plus(data, screened=True, stratum='number'):>8.2f} "
            f"{ier_plus(data, screened=True, stratum='common'):>7.2f}")
    return "\n".join(out)


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


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "evaluation/derived/full_v11"
    dirs = {name: base for name in SYSTEMS}
    print(summary_table(dirs, ci="--ci" in sys.argv))
    print()
    print(sensitivity_tables(dirs))
