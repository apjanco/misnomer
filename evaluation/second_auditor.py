"""Second-auditor robustness check: recompute Δ under an independent causal
LM (default GPT-2, 124M, WebText 2019 — a training pipeline unrelated to
Qwen2.5) for every filtered sieve-passing substitution, then report sign
agreement and the prior-strength ladder under the alternate auditor.

Usage:
    uv run python evaluation/second_auditor.py evaluation/derived/full_v11 [lm_model]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analysis import SIEVE, SYSTEMS, _collect_convergence, _raw_meta, load_system  # noqa: E402

from misnomer.aligner import tokenize_words  # noqa: E402
from misnomer.config import ScorerConfig  # noqa: E402
from misnomer.models.lm import LMScorer  # noqa: E402


def main() -> int:
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "evaluation/derived/full_v11")
    lm_name = sys.argv[2] if len(sys.argv) > 2 else "gpt2"

    cfg = ScorerConfig(scorer_version="1.1", lm_model=lm_name, device="auto",
                       torch_dtype="float32", allow_download=True, local_files_only=False)
    lm2 = LMScorer(cfg)
    assert lm2.is_transformer_backed, f"{lm_name} failed to load"
    print(f"second auditor: {lm2.resolved_model_name} on {lm2.resolved_device}")

    gt_suspect = _collect_convergence(base)
    gt_lines: dict[str, dict[int, str]] = {}

    agree = disagree = 0
    print(f"\n{'system':<22} {'Δ>0 share (2nd auditor)':>24}")
    for name, (raw, _) in SYSTEMS.items():
        if not (base / f"{raw}_rescored.jsonl").exists():
            continue
        if raw not in gt_lines:
            import json
            d = {}
            for line in open(Path(__file__).parent / f"{raw}_results.jsonl", encoding="utf-8"):
                r = json.loads(line)
                i = int(r.get("index", r.get("image_idx", r.get("image_path"))))
                d[i] = r.get("ground_truth") or ""
            gt_lines[raw] = d
        data = load_system(name, base, gt_suspect)
        plus = total = 0
        for ln in data.lines:
            if ln.excluded:
                continue
            subs, qwen = {}, {}
            for s in ln.substitutions:
                if s["error_type"] not in SIEVE or s.get("context_words") is None:
                    continue
                subs[s["context_words"]] = s["predicted_word"]
                qwen[s["context_words"]] = s["delta"]
            if not subs:
                continue
            words = tokenize_words(gt_lines[raw][ln.index], tokenizer=None)
            subs = {k: v for k, v in subs.items() if k < len(words)}
            deltas2 = lm2.substitution_deltas(words, subs)
            for pos, d2 in deltas2.items():
                if d2 is None or qwen.get(pos) is None:
                    continue
                total += 1
                if d2 > 0:
                    plus += 1
                if (d2 > 0) == (qwen[pos] > 0):
                    agree += 1
                else:
                    disagree += 1
        if total:
            print(f"{name:<22} {plus:>6}/{total:<6} = {plus/total:.0%}")

    if agree + disagree:
        print(f"\nsign agreement vs primary auditor: "
              f"{agree}/{agree+disagree} = {agree/(agree+disagree):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
