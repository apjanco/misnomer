# Evaluation data: provenance and status

Audited 2026-07-17 on branch `delta-v1.1`. Nothing in this directory is
modified by the rescoring pipeline; all derived outputs go to
`evaluation/derived/` (gitignored, regenerable).

## Source of truth: the seven per-model files

`<model>_results.jsonl`, one JSON object per IAM line. After deduplication
(below) every file covers exactly indices 0..10372 (10,373 lines), and
`ground_truth` is byte-identical across all seven files at every index
(verified — no misjoins).

Schema quirks from incremental development, all handled by
`rescore_iam.py`:

- The index key is `index` (chatgpt-gpt4o, chatgpt5, claude-vision,
  gemini-portkey), `image_idx` (chandra), or `image_path`
  (Qwen3-VL-8B-Instruct, tesseract — an integer despite the name).
- `cer`/`wer` scales differ across files (fractions vs percentages). The
  rescorer recomputes both and uses neither.
- The in-loop metric columns (`semantic_error`, `kl_divergence`, `entropy`,
  `mean_gt_rank`, `top5_accuracy`, `logprobs`, `mean_logprob`, `confidence`)
  are **invalid** (computed inside the generation loop; see the July 2026
  review). Never use them. All scoring is post-hoc via misnomer.

## Deduplication: last-write-wins, and why that is provably safe here

Two files contain duplicated indices from checkpoint-resume:
chatgpt5 (38 duplicated indices) and gemini-portkey (85). In **every**
duplicated index the earlier row has an empty `predicted_text` and the later
row has text: the run retried failed lines and appended the result. Zero
cases exist in the opposite direction, so keeping the last row per index
never discards a transcription. `rescore_iam.py` re-verifies this on every
run and aborts if the pattern is ever violated.

## Known data facts (post-dedupe)

- **Empty predictions** (scored as coverage failures, no word-level stats):
  chatgpt5 **2,285** (22.0% — real run behavior, spread across the corpus,
  median 20 s inference before returning nothing), tesseract 913,
  claude-vision 9, chatgpt-gpt4o 1 (+1 row with a recorded API 500),
  gemini-portkey 1.
- **Degenerate long outputs** (`len(pred) > 2·len(gt)+20`): gemini-portkey
  74 (mostly leaked chain-of-thought, e.g. index 8600 "Final check against
  constraints:** The user requested *only* the transcribed text..."),
  chatgpt-gpt4o 12, claude-vision 3, chandra 1 (a 6,532-character repetition
  loop against an 8-character line). The rescorer stores `pred_len`,
  `gt_len`, and `insertion_ratio` per row so analyses can filter these with
  a disclosed rule; no exclusion is baked into the data.
- IAM line 1222 genuinely begins with the word "sure" — it is a
  transcription, not a refusal preamble; misnomer's refusal regex is
  anchored and does not fire on it.

## Quarantined files — do not read, do not delete

- `full_eval_results.jsonl` (141,653 rows): self-appended composite;
  `image_path` no longer identifies rows for 5 of 7 models. Kept for
  history. Any analysis must use the per-model files instead.
- `false_negatives_analysis.jsonl` (37,547 rows): derivation unclear;
  treat as suspect until re-derived from the per-model files.

## Rescoring (misnomer v1.1, strict)

```bash
# smoke test: every 100th line, ~104 lines/model, ~1 min total on MPS
uv run python evaluation/rescore_iam.py --out evaluation/derived/smoke_v11 \
    --sample-every 100 --device auto --dtype float32

# full run: all 10,373 lines x 7 models, ~1-1.5 h on Apple silicon
uv run python evaluation/rescore_iam.py --out evaluation/derived/full_v11 \
    --device auto --dtype float32
```

Verified on the smoke sample: reruns are byte-identical (fp32); bfloat16
produces the same substitution sets with zero class flips and zero Δ sign
flips (Δ magnitudes drift, mean ≈0.4 — do not cite bf16 Δ to decimals), and
is ~4x slower than fp32 on MPS, so fp32 is the reference configuration.
Every run writes a `manifest.json` with config, package versions, input
file hashes, dedupe statistics, and timings.
