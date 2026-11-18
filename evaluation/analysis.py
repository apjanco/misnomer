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
from functools import lru_cache
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


# ---------------------------------------------------------------------------
# Error rates (CER / WER) on declared bases — misnomer.rates, "error-rates/1"
#
# Computed from the RAW model outputs, not from rescored files, so they do
# not depend on the auditor and need no rescoring. The prediction is used as
# the system returned it: preprocess()'s boilerplate stripping is NOT applied
# (on this corpus it removes parts of genuine transcriptions — lines that
# begin "Note ...", "Here are ...", "Following ..."); only its refusal test is
# used, to decide whether a line was answered at all.
#
# Views (all are printed or written; they answer different questions):
#   answered   the lines counted in the "Lines" column: not empty, not flagged
#              by the refusal test, not over-long (len > 2*len(reference)+20) —
#              the rule load_system applies to the scored files.
#   all lines  every reference line. A line the system failed on (empty,
#              refused, over-long) is scored against an EMPTY prediction, i.e.
#              as fully deleted, so one runaway output cannot outweigh a whole
#              page. A prediction flagged as a refusal whose REFERENCE also
#              reads like one ("I can't recall ...") is a transcription, not a
#              failure, and is scored as given.
#   ... excl. meta-output / ... meta-output as failures
#              the same two views with the META_OUTPUT lines below (leaked
#              reasoning, uncaught non-answers) taken out of the answered
#              lines, respectively scored as failures.
#
# Only this section needs the misnomer package; it is imported inside the
# functions so that the tables above keep running on the standard library.
# ---------------------------------------------------------------------------

N_LINES = 10373
REFERENCE_CONVENTION = "tokenised-english"  # the reference is tokenised LOB text
ERROR_RATE_BASES = ("verbatim", "content")
# first-pass file stem -> rows from a second pass over the lines left empty
BACKFILL = {"chatgpt5": "chatgpt5_backfilled_rows.jsonl"}
BACKFILL_NOTE = {
    "chatgpt5": "Second pass over the lines the first pass returned empty: same prompt and "
                "model id (gpt-5), larger max_completion_tokens (script default 4096 against "
                "1024; the value used was not recorded), OpenAI endpoint called directly "
                "instead of through the gateway. Model snapshot not recorded for either pass. "
                "Script: src/backfill_iam_gpt5.py in the not_flawless repository.",
}
# Lines column of the submitted table (all splits; PyLaia on the test split).
PUBLISHED_LINES = {
    "tesseract": 9460, "kraken-mccatmus": 10372, "pylaia-iam": 2915,
    "pylaia-iam-lm": 2915, "chatgpt-gpt4o": 10168, "chatgpt5": 10348,
    "chandra": 10369, "Qwen3-VL-8B-Instruct": 10370, "claude-vision": 10358,
    "gemini-portkey": 10295,
}
# Answered lines whose output is not a transcription at all: the model's own
# reasoning cut off mid-way (gemini), or a non-answer the refusal test does not
# catch (gpt-4o). Established by READING, on 2026-10-01: every answered line of
# every vision-language model with a content character error rate of 0.3 or
# more, plus every line carrying reasoning markers. Five of the gemini lines
# (749, 3364, 5075, 7100, 7416) contain the transcription next to the leaked
# text; they are listed all the same, because the output as returned is not
# one. The list only feeds the sensitivity views; the answered view and the
# Lines column are unaffected by it.
META_OUTPUT = {
    "gemini-portkey": (
        712, 725, 749, 1108, 1444, 1520, 1668, 2044, 2196, 2315, 2799, 3364, 3598, 3804,
        4875, 5075, 5330, 5824, 5826, 5949, 6177, 6527, 6629, 7025, 7100, 7321, 7416, 7468,
        7691, 7891, 8202, 8283, 8625, 8636, 8839, 9434, 9488, 9594, 9983, 9994,
    ),
    "chatgpt-gpt4o": (261, 1447, 1602, 10363),
}
VIEWS = ("answered", "all_lines", "answered_excl_meta_output", "all_lines_meta_output_failed")
FAILED = ("empty", "refusal", "degenerate")


@dataclass
class RawLine:
    index: int
    gt: str
    pred: str
    split: str | None
    backfilled: bool = False


def _row_index(rec: dict) -> int:
    return int(rec.get("index", rec.get("image_idx", rec.get("image_path"))))


def load_raw(name: str, merge_backfill: bool = True) -> tuple[dict[int, RawLine], dict]:
    """One system's raw outputs: deduplicated (last write wins by index), with
    a second-pass file merged in where one exists. Returns (lines, stats).

    The assumptions the merge rests on are checked here, not assumed: rows
    that share an index share one reference, a retried line never replaces
    text with nothing, indices cover 0..N_LINES-1, and a second-pass row only
    ever fills a line the first pass left empty, with the same reference.
    """
    raw, _ = SYSTEMS[name]
    by_index: dict[int, list[dict]] = defaultdict(list)
    for line in open(EVAL_DIR / f"{raw}_results.jsonl", encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            by_index[_row_index(rec)].append(rec)
    for i, recs in by_index.items():
        if len({r.get("ground_truth") or "" for r in recs}) != 1:
            raise ValueError(f"{raw}: rows for index {i} disagree on the reference")
        if not (recs[-1].get("predicted_text") or "").strip() and any(
                (r.get("predicted_text") or "").strip() for r in recs[:-1]):
            raise ValueError(f"{raw}: last-write-wins would drop text at index {i}")
    if set(by_index) != set(range(N_LINES)):
        raise ValueError(f"{raw}: index coverage is not 0..{N_LINES - 1}")
    lines = {
        i: RawLine(i, recs[-1].get("ground_truth") or "",
                   recs[-1].get("predicted_text") or "", recs[-1].get("split"))
        for i, recs in by_index.items()
    }
    stats = {
        "raw_rows": sum(len(r) for r in by_index.values()),
        "duplicated_indices": sum(1 for r in by_index.values() if len(r) > 1),
        "backfilled_rows": 0,
    }
    if merge_backfill and raw in BACKFILL:
        seen: set[int] = set()
        for line in open(EVAL_DIR / BACKFILL[raw], encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            i = _row_index(rec)
            if i in seen:
                raise ValueError(f"{BACKFILL[raw]}: index {i} occurs twice")
            seen.add(i)
            if lines[i].pred.strip():
                raise ValueError(f"{BACKFILL[raw]}: index {i} was not empty in the first pass")
            if (rec.get("ground_truth") or "") != lines[i].gt:
                raise ValueError(f"{BACKFILL[raw]}: reference differs at index {i}")
            lines[i] = RawLine(i, lines[i].gt, rec.get("predicted_text") or "",
                               lines[i].split, backfilled=True)
        stats["backfilled_rows"] = len(seen)
    return lines, stats


@lru_cache(maxsize=1)
def split_reference() -> dict[int, tuple[str, str]]:
    """index -> (IAM split, reference text), from the PyLaia result files —
    the only ones that record the split. The two files must agree."""
    first, _ = load_raw("pylaia-iam", merge_backfill=False)
    second, _ = load_raw("pylaia-iam-lm", merge_backfill=False)
    for i, ln in first.items():
        if ln.split is None or (second[i].split, second[i].gt) != (ln.split, ln.gt):
            raise ValueError(f"split or reference missing or inconsistent at index {i}")
    return {i: (ln.split, ln.gt) for i, ln in first.items()}


def split_map() -> dict[int, str]:
    """index -> IAM split ("train", "validation" or "test")."""
    return {i: split for i, (split, _) in split_reference().items()}


def line_status(pred: str, gt: str) -> str:
    """How a raw prediction counts, by the same tests as load_system.

    "empty", "refusal" and "degenerate" (over-long) are failures to answer;
    "ok" is counted as a transcription. "refusal-like-reference" is a
    prediction that trips the refusal test although the reference itself
    does too: a transcription, yet excluded from the answered lines by the
    rule the scored files use. The tests are those of the published table
    and are deliberately not improved here; what they let through is listed
    in META_OUTPUT.
    """
    from misnomer.preprocess import is_refusal

    if not pred.strip():
        return "empty"
    if is_refusal(pred):
        return "refusal-like-reference" if is_refusal(gt) else "refusal"
    if len(pred.strip()) > 2 * len(gt.strip()) + 20:
        return "degenerate"
    return "ok"


def _bootstrap_rate(edits: list[int], refs: list[int], n_boot: int,
                    seed: int) -> tuple[float, float] | None:
    """95% percentile interval of sum(edits)/sum(refs), resampling LINES with
    replacement. None when there is nothing to resample or no reference
    length. Lines are treated as independent, which they are not (a form is
    written by one hand), so the interval is somewhat too narrow."""
    import numpy as np

    e = np.asarray(edits, dtype=np.int64)
    r = np.asarray(refs, dtype=np.int64)
    if len(e) == 0 or r.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, len(e), len(e))
        denom = r[idx].sum()
        if denom:
            vals[b] = e[idx].sum() / denom
    if np.isnan(vals).all():
        return None
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _view(items: list, n_boot: int, seed: int) -> dict:
    """Pooled counts, rates and intervals of one view (zeros and nulls when
    the view holds no line)."""
    from misnomer.rates import pool

    entry: dict = {"lines": len(items)}
    pooled = pool(items) if items else None
    for basis in ERROR_RATE_BASES:
        if pooled is None:
            entry[basis] = {"char_edits": 0, "ref_chars": 0, "word_edits": 0, "ref_words": 0,
                            "cer": None, "wer": None}
            continue
        counts = getattr(pooled, basis)
        entry[basis] = counts.model_dump() | {"cer": counts.cer, "wer": counts.wer}
        if n_boot:
            per_line = [getattr(it, basis) for it in items]
            entry[basis]["cer_ci95"] = _bootstrap_rate(
                [c.char_edits for c in per_line], [c.ref_chars for c in per_line], n_boot, seed)
            entry[basis]["wer_ci95"] = _bootstrap_rate(
                [c.word_edits for c in per_line], [c.ref_words for c in per_line], n_boot, seed)
    return entry


def error_rate_rows(convention: str | None = REFERENCE_CONVENTION,
                    split: str | None = None, merge_backfill: bool = True,
                    n_boot: int = 1000, seed: int = 20261001) -> list[dict]:
    """Per system: line counts by status and pooled CER/WER counts on both
    bases, for every view.

    split=None scores each system on its own line set (all 10,373 lines;
    the PyLaia rows on the test split, because that model was trained on the
    rest). split="test" restricts EVERY system to the test split, which is
    the like-for-like basis for comparing anything with the PyLaia rows.
    Other splits are refused: they would score PyLaia on its training data.
    """
    from misnomer.conventions import resolve_convention
    from misnomer.rates import DEFINITION, UNICODE_FORM, error_rates

    if split not in (None, "test"):
        raise ValueError(f"split must be None or 'test', got {split!r}")
    resolved = resolve_convention(convention)
    reference = split_reference()
    rows = []
    for name, (_, test_only) in SYSTEMS.items():
        lines, stats = load_raw(name, merge_backfill)
        wanted = split or ("test" if test_only else None)
        meta = set(META_OUTPUT.get(name, ()))
        status_counts: dict[str, int] = defaultdict(int)
        views: dict[str, list] = {view: [] for view in VIEWS}
        n_backfilled = n_meta = 0
        for i in range(N_LINES):
            ln = lines[i]
            if ln.gt != reference[i][1]:
                raise ValueError(f"{name}: reference at index {i} differs from the split source")
            if wanted and reference[i][0] != wanted:
                continue
            status = line_status(ln.pred, ln.gt)
            if i in meta and status != "ok":
                raise ValueError(f"{name}: META_OUTPUT line {i} is not an answered line")
            status_counts[status] += 1
            n_backfilled += ln.backfilled
            deleted = None  # computed on demand: the line scored as fully deleted
            if status in FAILED:
                deleted = error_rates("", ln.gt, resolved)
                views["all_lines"].append(deleted)
                views["all_lines_meta_output_failed"].append(deleted)
                continue
            scored = error_rates(ln.pred, ln.gt, resolved)
            views["all_lines"].append(scored)
            if status == "ok":
                views["answered"].append(scored)
            if i in meta:
                n_meta += 1
                views["all_lines_meta_output_failed"].append(error_rates("", ln.gt, resolved))
            else:
                views["all_lines_meta_output_failed"].append(scored)
                if status == "ok":
                    views["answered_excl_meta_output"].append(scored)
        row = {
            "system": name, "line_set": wanted or "all",
            "lines": len(views["all_lines"]),
            "status": dict(status_counts),
            "meta_output_lines": n_meta,
            "backfilled_lines": n_backfilled,
            "load": stats,
            "definition": DEFINITION, "unicode_form": UNICODE_FORM,
            "convention": resolved.identity if resolved is not None else None,
            "views": {view: _view(items, n_boot, seed) for view, items in views.items()},
        }
        if split is None and merge_backfill and row["views"]["answered"]["lines"] != PUBLISHED_LINES[name]:
            raise AssertionError(
                f"{name}: {row['views']['answered']['lines']} answered lines, "
                f"published table has {PUBLISHED_LINES[name]}")
        rows.append(row)
    return rows


def _pct(value: float | None) -> str:
    return f"{100 * value:.2f}" if value is not None else "n/a"


def error_rate_table(rows: list[dict]) -> str:
    """CER and WER in per cent: both bases for the answered lines and for all
    lines, then the sensitivity to META_OUTPUT lines, then the legend."""
    line_sets = {r["line_set"] for r in rows}
    out = [
        f"line set: {line_sets.pop() if len(line_sets) == 1 else 'per system (PyLaia: test split)'}"
        f" | definition {rows[0]['definition']} | convention "
        f"{(rows[0]['convention'] or {}).get('name')} | rates in per cent",
        f"{'system':<22} {'lines':>6} {'answ.':>6} | {'ANSWERED content':>17} {'verbatim':>13} |"
        f" {'ALL LINES content':>18} {'verbatim':>13}",
        f"{'':<22} {'':>6} {'':>6} | {'CER':>8} {'WER':>8} {'CER':>6} {'WER':>6} |"
        f" {'CER':>9} {'WER':>8} {'CER':>6} {'WER':>6}",
    ]
    for r in rows:
        a, e = r["views"]["answered"], r["views"]["all_lines"]
        out.append(
            f"{r['system']:<22} {r['lines']:>6} {a['lines']:>6} | "
            f"{_pct(a['content']['cer']):>8} {_pct(a['content']['wer']):>8} "
            f"{_pct(a['verbatim']['cer']):>6} {_pct(a['verbatim']['wer']):>6} | "
            f"{_pct(e['content']['cer']):>9} {_pct(e['content']['wer']):>8} "
            f"{_pct(e['verbatim']['cer']):>6} {_pct(e['verbatim']['wer']):>6}")
    flagged = [r for r in rows if r["meta_output_lines"]]
    if flagged:
        out.append("")
        out.append("sensitivity, content basis: answered lines whose output is not a transcription "
                   "(leaked reasoning, uncaught non-answers; listed in META_OUTPUT)")
        out.append(f"{'system':<22} {'n':>4} | {'answered CER':>12} {'WER':>6}  "
                   f"{'without them':>12} {'WER':>6} | {'all lines CER':>13} {'WER':>6}  "
                   f"{'as failures':>12} {'WER':>6}")
        for r in flagged:
            v = r["views"]
            cells = [v[k]["content"] for k in VIEWS]
            out.append(
                f"{r['system']:<22} {r['meta_output_lines']:>4} | "
                f"{_pct(cells[0]['cer']):>12} {_pct(cells[0]['wer']):>6}  "
                f"{_pct(cells[2]['cer']):>12} {_pct(cells[2]['wer']):>6} | "
                f"{_pct(cells[1]['cer']):>13} {_pct(cells[1]['wer']):>6}  "
                f"{_pct(cells[3]['cer']):>12} {_pct(cells[3]['wer']):>6}")
    out.append("")
    out.append("ANSWERED: lines not empty, not flagged as a refusal, not over-long (the Lines column).")
    out.append("ALL LINES: every reference line; empty, refused and over-long outputs are scored as "
               "fully deleted (not on their own text).")
    out.append("content: case, punctuation at token edges and layout disregarded. verbatim: every "
               "character counts; it still carries the reference's detached single quotation marks, "
               "so it is not like-for-like between systems that copy that layout and systems that do not.")
    merged = [r for r in rows if r["backfilled_lines"]]
    for r in merged:
        out.append(f"{r['system']}: {r['backfilled_lines']} of these lines come from a second pass "
                   f"over lines the first pass left empty.")
    if "cer_ci95" in rows[0]["views"]["answered"]["content"]:
        out.append("95% intervals (line-level bootstrap; lines of one form are not independent, so "
                   "they are somewhat too narrow) are in the output file, not printed.")
    return "\n".join(out)


def _provenance(n_boot: int, seed: int) -> dict:
    """What identifies the code, data and settings behind a set of rows."""
    import hashlib
    import subprocess
    import sys
    import unicodedata
    from importlib import metadata, resources

    import misnomer.conventions
    import misnomer.rates

    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    repo = EVAL_DIR.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty: bool | None = bool(subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "evaluation/analysis.py",
             "pyproject.toml", "uv.lock"],
            cwd=repo, capture_output=True, text=True, check=True).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unknown", None

    def version(pkg: str) -> str:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return "not-installed"

    inputs = {f"{raw}_results.jsonl": sha((EVAL_DIR / f"{raw}_results.jsonl").read_bytes())
              for raw, _ in SYSTEMS.values()}
    inputs |= {f: sha((EVAL_DIR / f).read_bytes()) for f in BACKFILL.values()}
    return {
        "git_commit": commit,
        "git_dirty": dirty,  # uncommitted or untracked changes to the code; None = not a git checkout
        # the code that actually ran, whatever the git state says
        "code_sha256": {
            "misnomer/rates.py": sha(Path(misnomer.rates.__file__).read_bytes()),
            "misnomer/conventions.py": sha(Path(misnomer.conventions.__file__).read_bytes()),
            "misnomer/data/conventions.json": sha(
                resources.files("misnomer").joinpath("data/conventions.json").read_bytes()),
            "evaluation/analysis.py": sha(Path(__file__).read_bytes()),
        },
        "versions": {
            "python": sys.version.split()[0],
            "unicodedata": unicodedata.unidata_version,
            **{pkg: version(pkg) for pkg in ("misnomer", "rapidfuzz", "pydantic", "numpy")},
        },
        "prediction_input": "raw model output (no boilerplate stripping)",
        "intervals": {"method": "percentile bootstrap over lines, lines treated as independent; "
                                "the same resamples are used for CER and WER of a view",
                      "level": 0.95, "n_boot": n_boot, "seed": seed},
        "inputs_sha256": inputs,
        "second_pass": BACKFILL_NOTE,
        "meta_output": {name: list(indices) for name, indices in META_OUTPUT.items()},
    }


def write_error_rates(path: str | Path, tables: dict[str, list[dict]],
                      n_boot: int, seed: int) -> dict:
    """Write the rows together with what identifies how they were made: the
    commit and whether the code differs from it, hashes of the code that ran
    and of every input, package versions, the definition, the convention and
    the interval settings. Returns the payload."""
    import sys

    payload = _provenance(n_boot, seed) | {"tables": tables}
    if payload["git_dirty"] is not False:
        print("warning: the code differs from the recorded commit (or is not in a git checkout); "
              "the output identifies it by code_sha256 only", file=sys.stderr)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rates":
        # uv run python evaluation/analysis.py rates [--out FILE] [--no-ci]
        args = sys.argv[2:]
        out_path = None
        if "--out" in args:
            if args.index("--out") + 1 >= len(args):
                raise SystemExit("--out needs a file name")
            out_path = args[args.index("--out") + 1]
        n_boot, seed = (0 if "--no-ci" in args else 1000), 20261001
        tables = {
            "own_line_set": error_rate_rows(n_boot=n_boot, seed=seed),
            "test_split": error_rate_rows(split="test", n_boot=n_boot, seed=seed),
        }
        for rows in tables.values():
            print(error_rate_table(rows))
            print()
        if out_path:
            write_error_rates(out_path, tables, n_boot, seed)
        raise SystemExit(0)
    base = sys.argv[1] if len(sys.argv) > 1 else "evaluation/derived/full_v11"
    dirs = {name: base for name in SYSTEMS}
    print(summary_table(dirs, ci="--ci" in sys.argv))
    print()
    print(sensitivity_tables(dirs))
