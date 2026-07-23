# Validating the surprisal-contrast signal on the eval set

**Model:** `Qwen/Qwen2.5-0.5B` (text-only), the library's default `LMScorer`.
**Data:** `evaluation/full_eval_results.jsonl` (~141k VLM-OCR pages, 8 systems).
**Procedure:** word-aligned every errorful page with the library's own aligner,
extracted single-word `SUBSTITUTION`s where both gold and predicted are
alphabetic tokens, and scored the surprisal contrast

    Δ = s(pred) − s(gold),   override_risk = σ(−Δ)

with `s(w) = −log P(first-token(w) | gold left-context)` — both candidates
conditioned on the **same** gold left-context, so a shared rarity term cancels.

## What the run shows

1. **The signal fires on real overrides.** On genuine real-word→real-word
   swaps, the highest-override-risk pairs are exactly the target class —
   plausible, fluency-increasing misreads a character check forgives:
   `sited→noted`, `France→from`, `promptly→probably`, `sang→say`, `mild→wild`,
   `mark→work`, `twisting→hoisting`. About **21%** of real-word substitutions
   carry the override signature (Δ<0); ~14% are strong (risk>0.75).

2. **The contrast is a new axis, not a restatement of edit distance.**
   Spearman(edit distance, override_risk) ≈ **−0.15** on the clean set — nearly
   independent. This is the load-bearing result: the contrast adds information
   the existing `char_edit_distance` feature does not carry, so the composite
   gains discriminating power rather than double-counting.

## The honest limitation (and why the naive AUC is the wrong test)

A first pass computed an "override vs. character-corruption" AUC using **edit
distance as the class label** (small edit = char noise, large edit = override).
It came out at chance (~0.55 raw, ~0.40 on the clean subset). This is not the
signal failing — it is edit distance being an **invalid label** for the
override/garble distinction. That was the central finding of the original
diagnosis: *neither* edit distance *nor* cosine similarity separates these
classes, which is the entire reason a distributional contrast is needed.

Inspecting the raw (unfiltered) top-risk pairs made the contamination concrete:
`Anglesey→i`, `hiccup→is`, `Army→my`, `You→ofetceR` — OCR *fragments* that are
high-frequency tokens and so look like maximal overrides. A word-pair filter
that admits 1–2-character fragments turns them into false overrides. Restricting
to corpus-vocabulary words of length ≥3 (169 pairs) removes them and surfaces
the plausible-misread tail above.

**Consequence:** a rigorous precision/recall number for override detection
requires a *labeled* override set at word-pair granularity, which the eval does
not contain. The natural way to produce those labels is the LLM-reader tier
(`scripts/false_negatives/`), which adjudicates whether a swap changed meaning —
that is the recommended follow-on for turning this qualitative validation into a
calibrated one.

## Bottom line

The surprisal contrast behaves as designed on real data: it isolates
fluency-increasing real-word misreads, and it is close to orthogonal to edit
distance, so it is a genuine addition to the feature set rather than a
re-encoding of character noise. The correct next step is a reader-labeled subset
to calibrate `contrast_temperature` and set an operating threshold.

See `misnomer_contrast_validation.png` for the three-panel summary.
