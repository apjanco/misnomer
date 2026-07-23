<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="logo-light.png">
  <img width="100" src="logo-light.png" alt="Misnomer Logo">
</picture>

# misnomer

**Semantic error measurement for OCR/HTR output.**

Vision-Language Models used for OCR and HTR tend to substitute semantically plausible alternatives — synonyms, near-synonyms, contextually appropriate words — that pass human review undetected. Standard metrics (CER, WER) treat all substitutions equally. `misnomer` makes this class of error visible, measurable, and legible.

## Installation

```bash
pip install misnomer
```

To enable the transformer-backed perplexity scorer and sentence-transformer embedder:

```bash
pip install misnomer[model]
```

For multilingual collections:

```bash
pip install misnomer[model,multilingual]
```

## Quick start

```python
import misnomer

report = misnomer.score(
    predicted="The horse stood in the field near the chapel.",
    ground_truth="The house stood in the field near the chapel.",
)

print(report.scorer_mode)          # "full" | "standard" | "text_only"
print(report.document_score)       # float 0.0–1.0
print(report.semantic_error_count) # int
print(report.obvious_error_count)  # int

for ws in report.word_scores:
    if ws.error_type:
        print(ws.predicted_word, "→", ws.ground_truth_word,
              ws.error_type, f"({ws.composite_score:.2f})",
              f"risk={ws.override_risk:.2f}" if ws.override_risk is not None else "")
```

### Batch scoring

```python
reports = misnomer.score_batch([
    ("predicted_1", "ground_truth_1"),
    ("predicted_2", "ground_truth_2"),
])
```

### HTML highlighting

```python
html = misnomer.highlight(
    report,
    semantic_color="#FF6B6B",
    obvious_color="#FFD93D",
    threshold=0.3,
)
```

## CLI

```bash
# Score a single pair from files
misnomer score --predicted pred.txt --ground-truth gt.txt --output report.json

# Batch scoring from a CSV (columns: predicted, ground_truth)
misnomer score-batch --input pairs.csv --output reports.json

# Generate HTML highlighting
misnomer highlight --report report.json --output highlighted.html

# Show package and scorer version
misnomer version
```

## How it works

`misnomer` decouples transcription from evaluation. It treats both predicted and ground truth text as plain strings and applies its own fixed internal models:

| Signal | Method |
|--------|--------|
| **Surprisal contrast** | Causal LM forward pass (default: `Qwen/Qwen2.5-0.5B`) — under the *same* ground-truth left-context, does the model prefer the misread word over the correct word? The signed contrast `Δ = s(pred) − s(gold)` (full-word −logprob of each, summed over all of the word's tokens) is negative exactly when the language prior favours the wrong word, which is the signature of a language-prior override. See "The surprisal contrast" below. |
| **Semantic distance** | Sentence-transformer cosine similarity (default: `all-MiniLM-L6-v2`) — how close is the predicted word to the ground truth word? Words are encoded as `"A {word}."` to provide minimal sentence context; bare single-word encoding produces unreliable embeddings for short or rare words. |

These combine into a composite score per substituted word pair:

```
composite = 0.4 × override_risk + 0.6 × embedding_similarity
```

where `override_risk = σ(−Δ / T)` maps the signed contrast into `[0, 1]` (T is `contrast_temperature`, default 1.0). At `T = 1` this is exactly the model's relative preference for the wrong word, `P(pred|context) / (P(pred|context) + P(gold|context))`.

A high composite score indicates a concerning substitution: semantically close to the correct word (easy to miss) **and** one the model actively prefers over the truth in context (a plausible override, not random noise).

### The surprisal contrast

Earlier versions scored the *absolute* perplexity of the ground-truth word alone. That conflates two very different situations: a rare-but-correct word (surprising, but not an error) scores the same as a genuine override. Because the correct word is often the rarer one, absolute perplexity systematically flags correct rare entities as false positives.

The contrast fixes this by scoring **both** candidates under the identical left-context and taking the difference. A shared rarity term cancels, so a rare word read *correctly* contributes zero risk regardless of how surprising it is. Only when the model finds the *misread* word more fluent than the truth — the definition of a language-prior override — does the risk rise. On the project's own eval set this signal is nearly orthogonal to character edit distance (Spearman ≈ −0.15): it carries information the existing edit-distance feature does not.

Set `use_surprisal_contrast=False` in `ScorerConfig` to restore the legacy absolute-perplexity leg (`log(ppl) / log(1_000_000)`, clipped to `[0, 1]`).

### Error taxonomy

| Type | Description |
|------|-------------|
| `SEMANTIC` | Substitution with embedding similarity ≥ 0.35 to ground truth; likely to pass human review |
| `OBVIOUS` | Substitution with embedding similarity < 0.35; detectable on reading |
| `INSERTION` | Word present in prediction, absent in ground truth |
| `DELETION` | Word present in ground truth, absent in prediction |
| `MATCH` | Correct transcription |

> **Note on `text_only` mode:** when the sentence-transformer is unavailable, `error_type` is `None` for all substitutions and neither `semantic_error_count` nor `obvious_error_count` is incremented. The composite score is derived from the perplexity proxy alone.

### Operational tiers

`misnomer` degrades gracefully depending on installed dependencies:

| Mode | Condition | Perplexity source | Embedding source |
|------|-----------|-------------------|------------------|
| `full` | `transformers` + `torch` + `sentence-transformers` installed, model cached | Transformer LM (first token per word) | Sentence-transformer (frame-encoded) |
| `standard` | `transformers` + `torch` installed, no `sentence-transformers` | Transformer LM (first token per word) | None — `error_type` not assigned |
| `text_only` | No transformer dependencies | Frequency-based proxy | None — `error_type` not assigned |

The active mode is reported in `SemanticErrorReport.scorer_mode`. The `lm_model` and `embedder_model` fields in every report name what was *actually used* (e.g. `"frequency-proxy"`, `"rapidfuzz-char-similarity"`), not the configured model names.

## Scorer versioning

Scorer versions pin both model identities exactly so scores are comparable across projects and time:

| Version | LM | Embedder | Perplexity leg |
|---------|----|----------|----------------|
| `1.0` | `Qwen/Qwen2.5-0.5B` | `sentence-transformers/all-MiniLM-L6-v2` | absolute perplexity of gold word |
| `1.1` | `Qwen/Qwen2.5-0.5B` | `sentence-transformers/all-MiniLM-L6-v2` | signed full-word surprisal contrast → `override_risk` (default); also enables strict mode, tradition profiles, character anchoring, and per-substitution `delta` |

Specify a custom model for experimental use (disables the version guarantee):

```python
report = misnomer.score(
    predicted=...,
    ground_truth=...,
    config=misnomer.ScorerConfig(
        lm_model="gpt2",
        scorer_version="custom",
        allow_download=True,
    ),
)
```

## Development

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check src tests
uv run mypy src
```
