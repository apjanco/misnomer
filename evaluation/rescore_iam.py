"""Rescore the per-model IAM evaluation files under misnomer v1.1 (strict).

Reads the seven per-model ``evaluation/*_results.jsonl`` files (the source of
truth — ``full_eval_results.jsonl`` is quarantined, see README_DATA.md),
dedupes deterministically, and rescores every (predicted, ground_truth) pair
with the v1.1 gate classifier and Δ. Never modifies the input files.

Dedupe policy — last-write-wins by index. This is evidence-backed, not
assumed: every duplicated index in the raw files (38 in chatgpt5, 85 in
gemini-portkey) holds an EMPTY earlier row and a non-empty later row —
checkpoint-resume retried failed lines and appended the successful result.
Zero cases exist in the other direction.

Index key differs per file (Andy's incremental development): ``index``,
``image_idx``, or ``image_path`` (the latter holds the integer index despite
its name). All resolve to 0..10372, complete, with byte-identical
ground_truth across all seven models (verified 2026-07-17).

Rows whose prediction is empty or a refusal are still scored (they
short-circuit to ``is_refusal`` reports) and counted as coverage failures;
they contribute no word-level substitutions.

Outputs (per run directory; deterministic, no timestamps outside the manifest):
  <out>/<model>_rescored.jsonl   one line per IAM line: doc-level summary +
                                 full word scores for substitutions only
  <out>/substitutions.csv        flat cross-model table for inspection
  <out>/manifest.json            config, versions, hashes, dedupe stats, timing

Uses the scorer's internal seam (_score_with_models) so the auditor LM,
embedder, and dictionary load once for all seven models.

Example (smoke test — every 100th line, ~104 lines/model):
  uv run python evaluation/rescore_iam.py --out evaluation/derived/smoke_v11 \
      --sample-every 100 --device auto --dtype float32
Full run:
  uv run python evaluation/rescore_iam.py --out evaluation/derived/full_v11 \
      --device auto --dtype float32
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from importlib import metadata as importlib_metadata
from pathlib import Path

from misnomer.config import ScorerConfig
from misnomer.models.embedder import Embedder
from misnomer.models.lm import LMScorer
from misnomer.scorer import _build_dictionary, _enforce_strict, _score_with_models
from misnomer.traditions import resolve_tradition
from misnomer.unicode_ops import unicode_versions

EVAL_DIR = Path(__file__).parent
N_LINES = 10373  # IAM lines; every input file must dedupe to exactly this

MODELS = {
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct_results.jsonl",
    "chandra": "chandra_results.jsonl",
    "chatgpt-gpt4o": "chatgpt-gpt4o_results.jsonl",
    "chatgpt5": "chatgpt5_results.jsonl",
    "claude-vision": "claude-vision_results.jsonl",
    "gemini-portkey": "gemini-portkey_results.jsonl",
    "tesseract": "tesseract_results.jsonl",
    # conventional-ATR baseline added 2026-07-18: kraken McCATMuS_nfd_nofix_V1
    # (Zenodo 13788177) on the same Teklia/IAM-line images; see the run
    # script's docstring for the synthetic-baseline segmentation method.
    "kraken-mccatmus": "kraken-mccatmus_results.jsonl",
    # in-domain conventional baseline added 2026-07-18: Teklia/pylaia-iam
    # (trained on the IAM RWTH train split — cross-system comparisons must
    # restrict to rows with split == "test"; each row carries the split).
    "pylaia-iam": "pylaia-iam_results.jsonl",
}

INDEX_KEYS = ("index", "image_idx", "image_path")


def resolve_index(rec: dict) -> int:
    for key in INDEX_KEYS:
        if key in rec:
            return int(rec[key])
    raise KeyError(f"no index key among {INDEX_KEYS} in record with keys {sorted(rec)}")


def load_deduped(path: Path) -> tuple[dict[int, dict], dict]:
    """Load a per-model file, dedupe last-write-wins, verify completeness."""
    by_index: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_index[resolve_index(rec)].append(rec)

    dupes = {i: recs for i, recs in by_index.items() if len(recs) > 1}
    # Safety: the retry-append pattern must hold — never discard a non-empty
    # final row, and flag if last-write-wins would ever pick empty over text.
    regressions = [
        i for i, recs in dupes.items()
        if not (recs[-1].get("predicted_text") or "").strip()
        and any((r.get("predicted_text") or "").strip() for r in recs[:-1])
    ]
    if regressions:
        raise ValueError(
            f"{path.name}: last-write-wins would drop non-empty text at "
            f"indices {regressions[:10]} — dedupe policy assumption violated"
        )

    final = {i: recs[-1] for i, recs in by_index.items()}
    if set(final) != set(range(N_LINES)):
        raise ValueError(f"{path.name}: index coverage is not 0..{N_LINES - 1}")

    stats = {
        "raw_rows": sum(len(r) for r in by_index.values()),
        "unique_rows": len(final),
        "duplicated_indices": len(dupes),
        "empty_predictions": sum(
            1 for r in final.values() if not (r.get("predicted_text") or "").strip()
        ),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return final, stats


def pick_indices(args: argparse.Namespace) -> list[int]:
    indices = list(range(N_LINES))
    if args.sample_every:
        indices = indices[:: args.sample_every]
    if args.limit:
        indices = indices[: args.limit]
    return indices


def package_versions() -> dict[str, str]:
    versions = {}
    for pkg in ("torch", "transformers", "sentence-transformers", "pyspellchecker",
                "rapidfuzz", "pydantic", "Levenshtein"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=EVAL_DIR.parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


SUB_FIELDS = [
    "model", "index", "position", "ground_truth_word", "predicted_word",
    "error_type", "delta", "context_words", "embedding_similarity",
    "char_edit_distance",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="output directory for this run")
    parser.add_argument("--models", default="all",
                        help="comma-separated model names, or 'all'")
    parser.add_argument("--sample-every", type=int, default=None,
                        help="score every K-th index (systematic sample)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of indices (after sampling)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    model_names = list(MODELS) if args.models == "all" else args.models.split(",")
    unknown = [m for m in model_names if m not in MODELS]
    if unknown:
        parser.error(f"unknown models: {unknown}; known: {list(MODELS)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ScorerConfig(
        scorer_version="1.1",
        strict=True,
        device=args.device,
        torch_dtype=args.dtype,
    )
    print(f"loading auditor LM + embedder + dictionary (device={args.device}, "
          f"dtype={args.dtype}) ...", flush=True)
    lm = LMScorer(cfg)
    embedder = Embedder(cfg)
    dictionary = _build_dictionary(cfg)
    _enforce_strict(cfg, lm, embedder, dictionary)
    print(f"  lm={lm.resolved_model_name} on {lm.resolved_device}/{lm.resolved_dtype}; "
          f"embedder={embedder.resolved_model_name}", flush=True)

    indices = pick_indices(args)
    print(f"scoring {len(indices)} lines x {len(model_names)} models", flush=True)

    manifest: dict = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "config": cfg.model_dump() if hasattr(cfg, "model_dump") else vars(cfg),
        "resolved": {
            "lm": lm.resolved_model_name,
            "device": lm.resolved_device,
            "dtype": lm.resolved_dtype,
            "embedder": embedder.resolved_model_name,
            "tradition": resolve_tradition(cfg).name,
            "unicode": unicode_versions(),
        },
        "versions": package_versions(),
        "python": sys.version,
        "n_indices": len(indices),
        "sample_every": args.sample_every,
        "limit": args.limit,
        "inputs": {},
        "models": {},
    }

    sub_rows: list[dict] = []
    reference_gt: dict[int, str] | None = None

    for model in model_names:
        path = EVAL_DIR / MODELS[model]
        rows, stats = load_deduped(path)
        manifest["inputs"][model] = stats

        gt_map = {i: rows[i].get("ground_truth") or "" for i in indices}
        if reference_gt is None:
            reference_gt = gt_map
        elif gt_map != reference_gt:
            raise ValueError(f"{model}: ground_truth differs from previous models")

        out_path = out_dir / f"{model}_rescored.jsonl"
        t0 = time.monotonic()
        class_totals: Counter = Counter()
        n_refusal = 0
        n_subs = 0

        with out_path.open("w", encoding="utf-8") as out_fh:
            for n_done, i in enumerate(indices, 1):
                rec = rows[i]
                report = _score_with_models(
                    rec.get("predicted_text") or "",
                    rec.get("ground_truth") or "",
                    lm, embedder, cfg,
                    metadata={"index": i, "model": model},
                    dictionary=dictionary,
                )
                subs = [w for w in report.word_scores
                        if w.alignment_type == "SUBSTITUTION"]
                n_subs += len(subs)
                if report.is_refusal:
                    n_refusal += 1
                for key in ("semantic_error_count", "obvious_error_count",
                            "numeric_error_count", "normalization_count",
                            "archaizing_count", "garble_count", "homoglyph_count"):
                    class_totals[key] += getattr(report, key)

                position = {id(w): p for p, w in enumerate(report.word_scores)}
                for w in subs:
                    sub_rows.append({
                        "model": model, "index": i, "position": position[id(w)],
                        "ground_truth_word": w.ground_truth_word,
                        "predicted_word": w.predicted_word,
                        "error_type": w.error_type, "delta": w.delta,
                        "context_words": w.context_words,
                        "embedding_similarity": w.embedding_similarity,
                        "char_edit_distance": w.char_edit_distance,
                    })

                out_fh.write(json.dumps({
                    "index": i,
                    "model": model,
                    "is_refusal": report.is_refusal,
                    "preprocessing_applied": report.preprocessing_applied,
                    # evidence for analysis-time filtering of degenerate outputs
                    # (reasoning leaks, repetition loops) — no rule baked in here
                    "pred_len": len((rec.get("predicted_text") or "").strip()),
                    "gt_len": len((rec.get("ground_truth") or "").strip()),
                    "insertion_ratio": report.insertion_ratio,
                    "wer": report.wer,
                    "cer": report.cer,
                    "document_error_type": report.document_error_type,
                    "semantic_error_count": report.semantic_error_count,
                    "obvious_error_count": report.obvious_error_count,
                    "numeric_error_count": report.numeric_error_count,
                    "normalization_count": report.normalization_count,
                    "archaizing_count": report.archaizing_count,
                    "garble_count": report.garble_count,
                    "homoglyph_count": report.homoglyph_count,
                    "substitutions": [w.model_dump() for w in subs],
                }, ensure_ascii=False) + "\n")

                if n_done % args.progress_every == 0:
                    rate = n_done / (time.monotonic() - t0)
                    remaining = (len(indices) - n_done) / max(rate, 1e-9)
                    print(f"  {model}: {n_done}/{len(indices)} "
                          f"({rate:.1f} lines/s, ~{remaining / 60:.1f} min left)",
                          flush=True)

        elapsed = time.monotonic() - t0
        manifest["models"][model] = {
            "n_scored": len(indices),
            "n_refusal_or_empty": n_refusal,
            "n_substitutions": n_subs,
            "class_totals": dict(class_totals),
            "seconds": round(elapsed, 1),
            "lines_per_second": round(len(indices) / max(elapsed, 1e-9), 2),
        }
        print(f"{model}: {len(indices)} lines in {elapsed:.0f}s "
              f"({len(indices) / max(elapsed, 1e-9):.1f}/s), "
              f"{n_subs} substitutions, {n_refusal} empty/refusal", flush=True)

    with (out_dir / "substitutions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUB_FIELDS)
        writer.writeheader()
        writer.writerows(sub_rows)

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
