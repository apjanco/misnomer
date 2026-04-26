# Bug: Multi-token words get artificially deflated perplexity

**File:** `src/misnomer/models/lm.py` — `_perplexity_from_logprobs`, `_transformer_perplexities`  
**Severity:** Medium — affects correctness of the LM signal for any word that tokenizes into more than one sub-word piece

---

## Statement of the bug

`_perplexity_from_logprobs` computes:

```python
avg_logprob = sum(logprobs) / len(logprobs)
return math.exp(-avg_logprob)
```

For a word that tokenizes into N pieces, `logprobs` contains one value per token. The first token's log-probability reflects how surprising the word-start is in context (the signal we care about). Each subsequent token reflects how likely the *continuation of that specific word* is — a near-certain prediction, because the model already "knows" what word it's in. Averaging across all N tokens therefore pulls the per-token average log-probability toward zero (low surprisal), which collapses `exp(-avg)` toward 1.0.

**Measured evidence** (Qwen2.5-0.5B, context "The X stood in the field"):

| Word | Tokens | Perplexity |
|---|---|---|
| `horse` | 1 | 63,393 |
| `steed` | 2 | 5,207 |
| `cat` | 1 | 27,286 |
| `disambiguation` | 4 | 70 |
| `pharmaceutical` | 1 | 74,100 |

`"disambiguation"` (4 tokens, contextually bizarre) scores *lower* perplexity than `"horse"` (1 token, also contextually unexpected). The perplexity signal is anti-correlated with contextual rarity for multi-token words.

The effect scales with token count: a 4-token word needs 3 within-word continuations (each ~log-prob ≈ 0) to dilute the first real token's surprisal by a factor of 4.

---

## Why it matters

- Long or rare words (which tend to tokenize into many pieces) appear *more expected* to the scorer than short common words, inverting the intended signal.
- A genuine semantic substitution involving a multi-token synonym may score *lower* perplexity than the ground-truth word, reducing its composite score.
- The effect is silent: no warning is emitted, and the perplexity value looks plausible.

---

## Options for addressing it

### Option 1 — Use only the first token's log-probability (simplest)

Only the first token in a word span represents the true contextual prediction. Subsequent tokens are within-word continuations conditioned on already having started that word.

```python
@staticmethod
def _perplexity_from_logprobs(word: str, logprobs: list[float]) -> float:
    if not logprobs:
        return 1.0
    # Use only the first token: the prediction that reflects contextual surprise.
    return math.exp(-logprobs[0])
```

**Pros:** Eliminates the dilution entirely. Simple.  
**Cons:** Wastes information from continuation tokens. For very short common words (`"a"`, `"the"`) a single extremely low-surprisal token will dominate.

---

### Option 2 — Use only the first token per word, skip within-word continuations

Same as Option 1 but implemented at the span-gathering level: when iterating tokens over a word span, record only `token_log_probs[first_token_in_span]` and ignore the rest.

**Pros:** No change to `_perplexity_from_logprobs` signature; change is localized to the token-gathering loop in `_transformer_perplexities`.  
**Cons:** Same as Option 1.

---

### Option 3 — Take the minimum log-probability (most-surprising token)

```python
return math.exp(-min(logprobs))   # min log-prob = most surprising token
```

**Pros:** Robust; long words are scored by their hardest-to-predict token.  
**Cons:** Sensitive to tokenizer artifacts; a single unusual sub-word piece can dominate.

---

### Option 4 — Weight by token position within the word (discount continuations)

Apply exponentially decreasing weights so that the first token dominates:

```python
weights = [0.5 ** i for i in range(len(logprobs))]
w_sum = sum(weights)
avg_lp = sum(w * lp for w, lp in zip(weights, logprobs)) / w_sum
return math.exp(-avg_lp)
```

**Pros:** Principled; gracefully degrades to first-token weighting without discarding data.  
**Cons:** The decay rate (0.5 here) is a hyperparameter with no obvious grounding. Adds complexity.

---

### Option 5 — Use the raw sum of log-probabilities (joint probability, not average)

```python
return math.exp(-sum(logprobs))
```

This is the joint probability of the full word token sequence, which increases with word length even for common words. A 4-token word must predict all 4 tokens correctly, so joint probability is lower and perplexity higher.

**Pros:** Length-penalizes naturally without needing a per-character division.  
**Cons:** Makes perplexity uncomparable across words of different token counts (different effective ranges). A 1-token and a 4-token word with identical per-token log-probs will score very differently.

---

## Recommended approach

**Option 2** (first-token-only, at the span level) is the best balance of correctness and simplicity. The first token is the one the LM predicts purely from context, which is exactly what "contextual surprisal" means. Subsequent tokens are character-level continuations conditioned on the word already being chosen, and carrying them into the average introduces systematic length bias.

A follow-on improvement would be to add a test that verifies perplexity is not monotonically decreasing with token count for a set of words with known contextual rarity.
