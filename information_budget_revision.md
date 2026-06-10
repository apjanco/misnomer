# Information-Budget Scoring for Misnomer

### Implementation Specification v0.1 (Draft)

**Purpose:** Adapt the information-theoretic machinery from Chlon et al., *Predictable Compression Failures: Order Sensitivity and Information Budgeting for Evidence-Grounded Binary Adjudication* (arXiv:2509.11208) to Misnomer's word-level semantic error scoring.

**Scope:** Two implementable changes (Priority 1 and Priority 2) plus a design note on a longer-term third change (ISR-style gating).

---

## Background and Motivation

Misnomer's current perplexity signal answers the question: *how surprising is the ground truth word in context?* This is a proxy for error likelihood, but it is not a direct measure of how far the transcription model was from producing the correct word.

The EDFL framework in arXiv:2509.11208 provides a direct measure: the **information budget** — the minimum number of nats of additional signal that would have been required to shift the model's prediction to the correct outcome. The paper proves (Theorem 5) that for any binary event with prior probability q̄ and target probability p:

```
Δ̄ ≥ KL(Ber(p) ∥ Ber(q̄))
```

with the rare-event corollary that achieving reliability p = 1 − ε when q̄ ≪ 1 requires a budget that grows like log(1/q̄). In the paper's experiments, hallucination rate decreased approximately linearly with the information budget (≈ −12.7 percentage points per nat), supporting the interpretation of the budget as a calibrated risk quantity rather than an arbitrary score.

For Misnomer, the translation is:

| Paper concept | Misnomer analog |
|---|---|
| Bernoulli predicate g (correct/incorrect adjudication) | "Is this word transcribed correctly?" |
| Reference distribution P_ref (gold label, point mass) | Ground truth word (always available) |
| Model probability q for the correct outcome | LM probability of the ground truth token(s) in context |
| Information budget Δ̄ = KL(Ber(1) ∥ Ber(q)) | Per-word **bits-to-correct** score |
| RoH (Risk of Hallucination) | Document-level semantic error risk |

Because Misnomer's reference is a point mass (the ground truth word is known with certainty), the per-word KL budget simplifies:

```
KL(Ber(1) ∥ Ber(q)) = −log(q)
```

This means **the raw computation is nearly identical to what the scorer already does** — the change is in normalization, clipping, calibration, and interpretation, not in the forward pass.

---

## Priority 1 — Per-Word Information Budget (replaces normalized perplexity)

### 1.1 What changes

The `full` and `standard` scorer modes already run a causal LM forward pass over the ground truth text and extract per-token log-probabilities. Currently these are converted into a normalized perplexity term and combined as:

```
composite = 0.4 × normalized_perplexity + 0.6 × embedding_similarity
```

Replace the perplexity term with a clipped, normalized information budget.

### 1.2 Computation

For each substituted word pair (predicted_word, ground_truth_word) at aligned position i:

**Step 1 — Token probability of the ground truth word.**
Tokenize the ground truth word in context. If the word maps to multiple subword tokens t_1…t_k, the word-level probability is the product of conditional token probabilities (sum of log-probs):

```python
log_q = sum(token_logprobs)        # natural log, from the existing forward pass
q = math.exp(log_q)
```

**Step 2 — Smoothing.**
Guard against q = 0 (absolute continuity). The paper uses ε = 1e−9; adopt the same constant for cross-project comparability:

```python
EPSILON = 1e-9
q = max(q, EPSILON)
```

**Step 3 — Information budget in nats.**

```python
budget_nats = -math.log(q)         # == KL(Ber(1) || Ber(q)) for point-mass reference
```

**Step 4 — Symmetric clipping.**
Unbounded budgets (a single token with q ≈ ε contributes ~20.7 nats) destabilize document-level aggregates. The paper clips at B = 6 nats and reports robustness across B ∈ {4, 6, 8}. Adopt B = 6 as the default, exposed in `ScorerConfig`:

```python
CLIP_NATS = 6.0
budget_clipped = min(budget_nats, CLIP_NATS)
```

**Step 5 — Normalization for the composite.**
To keep the composite score in [0, 1], normalize by the clip ceiling:

```python
budget_normalized = budget_clipped / CLIP_NATS
```

**Step 6 — Composite score.**
Keep the existing weighting initially (this preserves comparability while the new signal is validated):

```python
composite = 0.4 * budget_normalized + 0.6 * embedding_similarity
```

### 1.3 Multi-token words: a necessary correction

Longer words naturally accumulate more subword tokens and therefore larger raw budgets, which would bias the score toward flagging long words. Two options, in order of preference:

1. **Per-token mean budget** — divide the word budget by token count k before clipping. This is the recommended default; it matches the paper's nats-per-token reporting convention (Jensen gaps in Experiment 1 are reported per token).
2. **Length-stratified normalization** — normalize against the empirical budget distribution for words of the same token count. More accurate but requires a calibration corpus; defer to a later version.

```python
budget_nats = -log_q / max(token_count, 1)   # per-token budget
```

### 1.4 Degradation behavior (text_only mode)

In `text_only` mode there is no LM forward pass; the current frequency-based proxy stands in for perplexity. The budget framing still applies: convert the frequency-based pseudo-probability to −log(q_freq) with the same clipping. Mark the result clearly:

- Add a field `budget_source: Literal["lm", "frequency_proxy"]` to `WordScore`.
- Document that frequency-proxy budgets are ordinal, not calibrated — they rank words usefully but the nat values are not comparable to LM-derived budgets.

### 1.5 API changes

```python
@dataclass
class WordScore:
    # existing fields...
    budget_nats: float            # clipped, per-token information budget
    budget_source: str            # "lm" | "frequency_proxy"
    # normalized_perplexity retained for one deprecation cycle, marked deprecated

@dataclass
class ScorerConfig:
    # existing fields...
    clip_nats: float = 6.0
    epsilon: float = 1e-9
```

### 1.6 Versioning

This changes score semantics. Bump the scorer version (e.g., `1.0` → `2.0`) in the version table so that scores are never silently compared across the boundary. Keep `1.0` reproducible behind a config flag for one release if any downstream project depends on it.

### 1.7 Validation before merging

- **Rank correlation check:** on an existing labeled set, Spearman correlation between old composite and new composite should be high (> 0.9 expected, since the transform is monotone in −log q for single-token words). Large divergences will concentrate in multi-token words — inspect those manually; they are where the correction matters.
- **Known-error recovery:** verify that previously flagged SEMANTIC errors are still flagged at the same threshold, and document any boundary cases that flip.
- **Long-word bias test:** confirm the per-token correction removes the correlation between word length (in tokens) and budget that exists in the uncorrected version.

---

## Priority 2 — Document-Level Risk Score (RoH)

### 2.1 What this adds

A calibrated document-level statement of semantic error risk, derived from the paper's Risk-of-Hallucination planner:

```
RoH(x) = 1 − p_max(Δ̄(x), q̄(x))
```

where p_max(Δ̄, q̄) is the largest achievable success probability given an observed average budget Δ̄ and prior q̄ — i.e., the largest p satisfying KL(Ber(p) ∥ Ber(q̄)) ≤ Δ̄. This replaces "this document contains 14 errors" with "this document carries an estimated X% risk of undetected semantic substitution."

### 2.2 Computation

**Inputs (all already computed per word after Priority 1):**

- For every aligned word position (not just substitutions), the LM probability q_i of the ground truth word. Substitution positions and match positions both carry information: a document full of high-budget *matches* is a document where the transcription model got lucky, and risk should reflect that.

**Step 1 — Document prior and average budget.**

```python
q_bar  = mean(q_i for all aligned positions)                  # mean prob of correct word
delta_bar = mean(clip(-log(q_i)/k_i, CLIP_NATS) for all positions)  # mean per-token budget
```

**Step 2 — Solve for p_max.**
p_max is the unique p ∈ [q̄, 1) with KL(Ber(p) ∥ Ber(q̄)) = Δ̄ (or 1.0 if no such p exists below 1). The KL is monotonically increasing in p on [q̄, 1), so a bisection solve is sufficient and dependency-free:

```python
def bernoulli_kl(p: float, q: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    q = min(max(q, 1e-12), 1 - 1e-12)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def p_max(delta_bar: float, q_bar: float, tol: float = 1e-9) -> float:
    lo, hi = q_bar, 1.0 - 1e-12
    if bernoulli_kl(hi, q_bar) <= delta_bar:
        return hi
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if bernoulli_kl(mid, q_bar) <= delta_bar:
            lo = mid
        else:
            hi = mid
    return lo

roh = 1.0 - p_max(delta_bar, q_bar)
```

**Step 3 — Report.**

```python
@dataclass
class SemanticErrorReport:
    # existing fields...
    document_roh: float        # 0.0–1.0; estimated risk of undetected semantic substitution
    roh_inputs: RoHInputs      # delta_bar, q_bar, n_positions — for auditability
```

### 2.3 Interpretation guidance (include in README)

- RoH is **relative to the internal scorer's view of the text**, not an absolute probability of error — it inherits the LM's calibration. State this plainly.
- RoH is most useful **comparatively**: sorting a batch of documents by RoH for review triage is its primary intended use. The `score_batch` output should support sorting on this field.
- Because match positions contribute, two documents with identical error counts can have very different RoH — that is the point, not a bug.

### 2.4 Validation

- On a corpus with known ground truth, bin documents by RoH decile and verify that the realized semantic-error density increases monotonically across bins. This is the document-level analog of a calibration curve and is the single most persuasive figure for a paper or grant narrative about the tool.
- Check stability of RoH under the clipping parameter (B ∈ {4, 6, 8}), mirroring the robustness appendix of the source paper.

---

## Longer-Term — ISR-Style Analytic Threshold (design note, not yet implementable)

### The idea

The current flagging threshold (and the 0.4/0.6 composite weighting) is an empirical design choice. The paper's Information Sufficiency Ratio replaces tuned thresholds with an analytic gate: compare the observed budget Δ̄ against the minimum budget B2T required to reach a user-specified target reliability p* = 1 − h*, and act when ISR = Δ̄/B2T ≥ 1. The threshold (1.0) is fixed by theory; only the *target* (h*) is user-specified. For Misnomer this would mean: a user states their tolerated false-positive rate, and the per-word flagging boundary follows analytically — different for archival letters than for clinical notes, with no retuning.

### Why it is not a drop-in

1. **The predicate direction is inverted.** The paper gates *answer vs. abstain* on the model's own output. Misnomer gates *flag vs. don't flag* on a comparison against ground truth. The mapping of (p*, q_lo, Δ̄) onto Misnomer's flagging decision needs to be worked out carefully — in particular, what q_lo (the paper's worst-case permutation probability) corresponds to when there is no permutation ensemble.
2. **The dispersion machinery has no current analog.** The paper's q_lo comes from scoring under m permutations of evidence. The natural Misnomer analog is scoring each word under multiple **context conditions** (sentence-only / paragraph / full preceding text), with q_lo as the minimum across conditions. This requires 2–3× forward passes and changes the scorer's compute profile.
3. **Calibration of the embedding signal.** ISR operates purely on Bernoulli probabilities. Misnomer's second signal (embedding similarity) is not a probability. Either it must be calibrated into one (e.g., via isotonic regression on a labeled set, mapping similarity → P(passes human review)), or the ISR gate applies only to the budget axis with embedding similarity kept as an orthogonal classifier of SEMANTIC vs. OBVIOUS.

### Suggested next steps, in order

1. **Measure context dispersion first.** Before building anything, run the existing scorer under 3 context-window conditions on an existing evaluation corpus and measure per-word dispersion in −log q. If dispersion is small for the words Misnomer cares about, the multi-condition machinery isn't warranted and a single-pass ISR variant (q_lo := q) suffices. If dispersion is large, that is itself a publishable finding about OCR/HTR evaluation, and it motivates the full design. The paper predicts dispersion growing like a + b·log(context length) — checking whether transcription scoring shows the same law is a cheap, well-posed experiment.
2. **Calibrate the embedding signal on labeled review data.** Collect or reuse a set of substitutions labeled by whether human reviewers caught them; fit similarity → P(missed by review). This is needed regardless of the ISR work and improves the SEMANTIC/OBVIOUS boundary on its own.
3. **Prototype the gate behind a config flag.** `ScorerConfig(target_precision=0.95)` → derive the per-word flagging boundary from B2T; compare flag sets against the current threshold on the evaluation corpus; report precision/recall at matched coverage.
4. **Decide on the composite's future.** If the calibrated two-signal version works, the 0.4/0.6 weighted sum can be retired in favor of a probabilistic combination — at which point the scorer version bumps again, and the spec for v3 gets written.

### What to read

- arXiv:2509.11208, Sections 3.4–3.5 (EDFL, Algorithm 1, Box 1) and Appendix B.3 (worked numerical example) — the minimal set needed to implement the gate.
- Experiment 2 (causal dose-response) for the empirical justification of treating the budget as a risk quantity.
- The paper's Limitations section, which flags that guarantees are relative to the reference distribution — for Misnomer this means guarantees are relative to the internal LM's calibration, which should be stated wherever ISR-derived numbers are surfaced.

---

## Summary of Changes by Release

| Release | Change | API impact | Scorer version |
|---|---|---|---|
| Next minor | Per-word information budget replaces normalized perplexity; per-token correction; ε-smoothing; B = 6 clipping | `WordScore.budget_nats`, `budget_source`; `ScorerConfig.clip_nats`, `epsilon` | 2.0 |
| Same or following minor | Document-level RoH | `SemanticErrorReport.document_roh`, `roh_inputs`; batch sorting | 2.0 |
| Future (after dispersion study) | ISR analytic flagging gate; calibrated embedding probability | `ScorerConfig.target_precision`; possible retirement of fixed composite weights | 3.0 |

## Reference

Chlon, L., Karim, A., Chlon, M., & Awada, M. (2026). *Predictable Compression Failures: Order Sensitivity and Information Budgeting for Evidence-Grounded Binary Adjudication.* arXiv:2509.11208 [stat.ML].
