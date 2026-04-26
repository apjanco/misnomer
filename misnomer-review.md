# `misnomer` code review — actionable fixes

Repository: [`apjanco/misnomer`](https://github.com/apjanco/misnomer) (v0.1.0, single commit)
Scope: does the code support its advertised claim — calculating perplexity and producing a semantic error metric for OCR/HTR output?

**Summary:** Yes, at a structural level — alignment, a two-signal scorer (perplexity + embedding similarity), tiered fallback, a Pydantic report, and a CLI all exist and roughly do what the README says. But several substantive problems mean the metric doesn't mean what it claims to mean: one outright spec/code contradiction, a "perplexity" that isn't perplexity in the standard sense, a fallback tier that inverts the semantic/obvious distinction, a silently-lying `lm_model` report field, and a frequency-weighting feature that is largely inert on realistic inputs. Test coverage is thin enough that none of this is caught.

The issues below are ordered by severity. Each has a concrete fix and a suggested test.

---

## Priority 1 — Critical correctness issues

### 1.1 `text_only` and `standard` fallback tiers invert the semantic/obvious distinction

**Files:** `src/misnomer/models/embedder.py`, `src/misnomer/classifier.py`, `src/misnomer/composite.py`, `src/misnomer/scorer.py`

**Problem.** When `sentence-transformers` is unavailable, `Embedder.similarity()` falls back to `rapidfuzz.fuzz.ratio`, which is *character* similarity — a string-edit score. The downstream classifier and composite score treat this value as if it were *semantic* similarity.

Concrete consequence on the spec's own worked example:

- `horse → house` (the spec's canonical `OBVIOUS` error) has character similarity ≈ 0.8 → classified `SEMANTIC`, high composite score.
- `horse → steed` (the spec's canonical `SEMANTIC` error) has character similarity ≈ 0.0 → classified `OBVIOUS`, low composite score.

The taxonomy is flipped in both fallback tiers. The spec in section 7 claims the `text_only` mode is "weaker but still classifies SEMANTIC vs OBVIOUS errors and produces a meaningful composite score." It does not — it produces an inverted one.

**Fix (choose one).**

1. **Recommended:** In fallback mode, do not claim to produce a semantic classification. Set `error_type = None` for substitutions, set `embedding_similarity` to a sentinel (e.g., `nan` or a documented placeholder), and fold only the perplexity signal into the composite. The `scorer_mode` field already tells the user what mode they're in; the report should behave consistently with that mode.
2. In fallback mode, explicitly convert char similarity into dissimilarity (`1 - ratio/100`) and rename the reported field so the user isn't misled — but this only papers over the problem because character dissimilarity still isn't semantic similarity.

**Suggested test.**

```python
def test_text_only_does_not_misclassify_obvious_error():
    # Force fallback by using a model that won't load.
    cfg = ScorerConfig(lm_model="does-not-exist", local_files_only=True)
    report = score("The horse stood", "The house stood", config=cfg)
    assert report.scorer_mode == "text_only"
    sub = next(ws for ws in report.word_scores if ws.alignment_type == "SUBSTITUTION")
    # horse→house is character-similar but semantically distant.
    # Assert the scorer does NOT claim this is a SEMANTIC (undetectable) error.
    assert sub.error_type != "SEMANTIC"
```

---

### 1.2 README and the committed spec define the composite score with opposite signs

**Files:** `semantic_error_spec.md` §6.4, `README.md`, `src/misnomer/composite.py`

**Problem.** The checked-in spec says:

> `composite = w1 * normalized_perplexity + w2 * (1 - embedding_similarity)`

The README says:

> `composite = 0.4 × normalized_perplexity + 0.6 × embedding_similarity`

The code (`composite.py:20-21`) implements the README version. The two are *opposite*, not equivalent. Any future reader of both docs is guaranteed to be confused about which is authoritative.

**Fix.** The README's reading matches the project's own problem statement (errors that pass human review — i.e., high similarity — are the dangerous ones). Update `semantic_error_spec.md` §6.4 to match the code:

```
composite = w1 * normalized_perplexity + w2 * embedding_similarity

A high composite indicates a concerning substitution: semantically close to
the correct word (likely to pass human review) AND surprising to the LM
(contextually unlikely, so not a plausible correction).
```

Also update section 6.3 of the spec — it currently says "High similarity (close to 1.0) = semantic error candidate" in one place but describes the composite in the opposite way in section 6.4. Make both sections consistent.

---

### 1.3 "Perplexity" is not perplexity as the word is normally defined

**File:** `src/misnomer/models/lm.py` (lines 107–128)

**Problem.** The standard definition is
$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_i \log p(x_i \mid x_{<i})\right)$$
The code computes:

```python
avg_logprob = sum(logprobs) / len(logprobs)      # per-token avg log-prob
avg_logprob_per_char = avg_logprob / char_len    # divided AGAIN by chars
return math.exp(-avg_logprob_per_char)
```

The per-token average is divided a second time by character length, then exponentiated. This exponent has nonstandard units (log-prob per token per character) and collapses toward 1.0 for long words regardless of actual surprisal. The per-word values are therefore not comparable across words of different lengths, and the `WordScore.perplexity` field — which is exposed to users as if it meant something standard — does not.

The spec in §6.2 describes this formula literally, so the code faithfully implements the spec — but the spec itself is wrong to call the result perplexity.

**Fix.** Compute standard per-token perplexity:

```python
@staticmethod
def _perplexity_from_logprobs(word: str, logprobs: list[float]) -> float:
    if not logprobs:
        return 1.0
    avg_logprob = sum(logprobs) / len(logprobs)
    return math.exp(-avg_logprob)
```

If length-normalized surprisal is desired for the composite, compute it as a *separate* value and expose it under a separate field name (e.g., `length_normalized_surprisal`). Update `semantic_error_spec.md` §6.2 accordingly.

**Related, smaller issue in the same file.** `_transformer_perplexities` (line 60) calls the tokenizer without `add_special_tokens=False` and then unconditionally skips position 0 as if it were always BOS. Qwen2.5's tokenizer does *not* add BOS by default, so this silently discards the first real token's log-probability. Either pass `add_special_tokens=False` explicitly and start from position 0, or detect BOS presence from the tokenizer's special-tokens map and skip conditionally.

**Suggested test.**

```python
def test_perplexity_is_standard_per_token():
    # Pin behavior for a known input once the fix is in.
    # The exact value is less important than: two short words with
    # identical per-token log-probs should get identical perplexity,
    # regardless of character length.
    # (Current behavior: they differ by char-length ratio.)
    ...
```

---

## Priority 2 — Report honesty and consistency

### 2.1 `lm_model` in the report lies when the transformer silently fails

**File:** `src/misnomer/models/lm.py` (lines 23–39), `src/misnomer/scorer.py` (line 152)

**Problem.** `LMScorer.__init__` wraps model loading in `except Exception: ...` with no logging. On a fresh machine with `local_files_only=True` (the default) and no cached weights, loading will fail and the scorer falls through to `_proxy_perplexities`. The resulting report correctly sets `scorer_mode="text_only"` — but `lm_model` is still populated from `cfg.lm_model` and reads `"Qwen/Qwen2.5-0.5B"`, suggesting the Qwen model was used when it wasn't.

A user inspecting the JSON report would have no way to know the scores came from the frequency proxy.

**Fix.**

1. In `LMScorer.__init__`, log the exception (at minimum via `logging.warning`) rather than silently swallowing it — blanket `except Exception` also masks genuine bugs during development.
2. Add a `LMScorer.resolved_model_name` property that returns `self.model_name` when the transformer loaded and `"frequency-proxy"` (or similar sentinel) when it didn't.
3. In `scorer.py` line 152, use `lm.resolved_model_name` instead of `cfg.lm_model` when populating the report.

Do the equivalent for `Embedder`: add `resolved_model_name` that returns `"rapidfuzz-char-similarity"` when the sentence-transformer isn't loaded.

---

### 2.2 Tier detection reaches into a private attribute

**File:** `src/misnomer/scorer.py` (line 36)

**Problem.**

```python
if lm.is_transformer_backed and embedder._model is not None:
    scorer_mode: ScorerMode = "full"
```

`lm.is_transformer_backed` is a public property; `embedder._model is not None` reads a private attribute directly. Minor, but easy to fix.

**Fix.** Add `Embedder.is_model_backed` as a public property mirroring `LMScorer.is_transformer_backed`, and use both in `scorer.py`.

---

## Priority 3 — Correctness bugs that happen to not matter much today

### 3.1 Document-score frequency weighting is largely inert

**File:** `src/misnomer/scorer.py` (lines 12–14, 102–104)

**Problem.** The spec §6.5 promises "weighted by word frequency rank (rare words weighted higher)." The code weights by `1 / count_in_this_document`. In a 20-word OCR snippet where every word is unique (the common case), every substitution gets weight 1.0 — i.e., the feature does nothing. In a longer document where "the" appears 10 times, "the" gets weight 0.1 — but that's because it's frequent *in this document*, not because it's common in English. A unigram from a reference corpus is what the spec actually promises.

**Fix options.**

- **Simple:** ship a small frozen unigram frequency table (e.g., from wordfreq) and use that for weighting. Document the source in the spec.
- **Simpler:** drop the per-document-count weighting, weight all substitutions equally, and update §6.5 of the spec to match. Don't claim a feature the code can't deliver.

### 3.2 `score_batch` applies the same `metadata` to every report

**File:** `src/misnomer/scorer.py` (lines 164–180)

**Problem.** The batch function takes a single `metadata` dict and stamps it onto every report. A user scoring 1,000 document pairs almost certainly wants per-pair metadata (page number, source filename, etc.).

**Fix.** Change the signature to accept either `list[tuple[str, str, dict]]` triples or a parallel `metadata_list: list[dict] | None`.

### 3.3 `normalize_perplexities` silently zeroes out the perplexity signal

**File:** `src/misnomer/composite.py` (lines 4–11)

**Problem.** When all substitutions have identical raw perplexity (happens in the proxy path for documents where all substituted words have the same count and length), `high == low` and the function returns all zeros. The perplexity term then contributes zero to every composite score, silently. No warning; no fallback.

**Fix.** When the range is degenerate, either return `[0.5] * len(perplexities)` (neutral) or expose a flag on the report that the perplexity signal was unusable for this document.

### 3.4 Alignment/substitution indexing is correct but fragile

**File:** `src/misnomer/scorer.py` (lines 44–54, 82–88)

**Problem.** Two separate passes build `substitution_gt_indices` and then consume it via `subst_cursor`. Any future change to which alignment types advance `gt_idx` will silently desync the cursor from the perplexities list. No test guards this invariant.

**Fix.** Compute the substitution's perplexity at the site where each `SUBSTITUTION` item is emitted, not across two passes. Reduces surface area for drift.

### 3.5 Empty broad exception handling masks real bugs

**Files:** `src/misnomer/models/lm.py` (line 36), `src/misnomer/models/embedder.py` (line 18)

**Problem.** Both classes wrap their entire initialization in `except Exception: self._model = None`. That catches network errors, disk-full errors, genuine code errors introduced during refactors, and the expected "weights not cached" case — all equivalently. During development this will eat real bugs.

**Fix.** Narrow to the specific exceptions that are expected (`ImportError`, `OSError`, `huggingface_hub.errors.LocalEntryNotFoundError`), and at minimum log the exception type and message via `logging.warning` so a debugging user has a trail.

---

## Priority 4 — Test coverage

**File:** `tests/`

Three test files; `test_scorer.py` has two tests totaling eight lines. There are no tests for:

- the composite score (is it actually higher for semantic than obvious errors?)
- the transformer-backed perplexity path
- tier detection
- the `score_batch` happy path
- the CLI (any command)
- the HTML highlighter
- any fixture of the spec's worked examples (`horse→steed` vs `horse→house`)

The spec's own "reproducibility via fixed scorer versions" claim has no teeth without tests that pin actual scores for canonical inputs.

**Suggested minimum additions.**

1. Fixture test pinning composite scores for `horse→steed` vs `horse→house` on scorer v1.0 in `full` mode (skippable if models not cached).
2. Fixture test that `horse→house` is `OBVIOUS` and `horse→steed` is `SEMANTIC` in `full` mode.
3. Regression test for 1.1 above: fallback tiers do not misclassify `horse→house` as `SEMANTIC`.
4. Test that `score_batch` produces `len(pairs)` reports, each independently constructed.
5. Smoke test invoking each CLI subcommand on a tmp-path fixture via `typer.testing.CliRunner`.
6. Round-trip test: `score(...) → model_dump_json → SemanticErrorReport.model_validate_json → highlight(...)`.

---

## Priority 5 — Small stuff

- `LMScorer.word_perplexities` returns `dict[int, float]` but is only ever indexed by sequential integer. Use `list[float]`.
- `Embedder.similarity` re-encodes two strings per call. For documents with many substitutions, batch-encode unique words once and look them up.
- `composite_score` clips to `[0, 1]` — a defensive no-op given the inputs are already bounded and the weights sum to 1. Fine to keep, but worth a comment saying it's defensive.
- `scorer.py` shadows the outer `score` function with a local `score_value` at line 90; fine, but the word "score" is doing four different jobs in this module (function name, field name, value, package name).
- `aligner.py` is a textbook WER/Needleman–Wunsch DP — correct, no changes needed, but worth a docstring citing that it's the same algorithm used for WER, so future maintainers don't refactor it into something else.
- `pyproject.toml` pins `numpy>=2.4.4`, `pydantic>=2.13.3`, and `transformers>=5.5.4` — all unusually high minimums that will exclude many existing environments. Worth checking whether these floors are actually required.
- Docstrings on public functions (`score`, `score_batch`, `highlight`) would help — currently they have none.

---

## Spec-vs-code contradiction summary

| Spec says | Code does | Action |
|---|---|---|
| `composite = w1·normalized_perplexity + w2·(1 - embedding_similarity)` | `composite = 0.4·normalized_perplexity + 0.6·embedding_similarity` | Update spec §6.4 to match code |
| Perplexity per word = `exp(-avg_logprob_per_char)` | Implements exactly this | Both are wrong — fix code and spec to standard `exp(-avg_logprob)` |
| Text-only mode "classifies SEMANTIC vs OBVIOUS and produces a meaningful composite score" | Inverts the classification | Fix code behavior, not the spec |
| Document score "weighted by word frequency rank" | Weighted by in-document count | Either fix code to use corpus frequency, or update spec |
| Scorer version pins LM and embedder "exactly" for reproducibility | `lm_model` still populated from config when fallback fires | Fix report to name what was actually used |

---

## Recommended order of work

1. Fix **1.3** (perplexity formula) and **1.2** (spec/README contradiction) together — both touch the definition of the metric and should land as one coherent change with updated spec.
2. Fix **1.1** (fallback inverts taxonomy) next — user-visible and changes the meaning of every `text_only` report.
3. Fix **2.1** (lying `lm_model` field) and **2.2** (private attribute access) — small, improves trust in every report.
4. Priority 3 items as opportunity allows.
5. Land the test fixtures from Priority 4 alongside each of the above so the fixes are pinned.
