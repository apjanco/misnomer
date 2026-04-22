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
              ws.error_type, f"({ws.composite_score:.2f})")
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
| **Perplexity** | Causal LM forward pass (default: `Qwen/Qwen2.5-0.5B`) — how surprising is the ground truth word in context? |
| **Semantic distance** | Sentence-transformer cosine similarity (default: `all-MiniLM-L6-v2`) — how close is the predicted word to the ground truth word? |

These combine into a composite score per substituted word pair:

```
composite = 0.4 × normalized_perplexity + 0.6 × embedding_similarity
```

A high score indicates a concerning substitution: semantically close to the correct word (easy to miss) and surprising to the LM (contextually unlikely to appear by chance).

### Error taxonomy

| Type | Description |
|------|-------------|
| `SEMANTIC` | Substitution with high embedding similarity to ground truth; likely to pass human review |
| `OBVIOUS` | Substitution with low embedding similarity; detectable on reading |
| `INSERTION` | Word present in prediction, absent in ground truth |
| `DELETION` | Word present in ground truth, absent in prediction |
| `MATCH` | Correct transcription |

### Operational tiers

`misnomer` degrades gracefully depending on installed dependencies:

| Mode | Condition | Perplexity source |
|------|-----------|-------------------|
| `full` | `transformers` + `torch` + `sentence-transformers` installed, model cached | Transformer LM forward pass |
| `standard` | `transformers` + `torch` installed, no `sentence-transformers` | Transformer LM; embedding falls back to char similarity |
| `text_only` | No transformer dependencies | Frequency-based proxy; embedding falls back to char similarity |

The active mode is reported in `SemanticErrorReport.scorer_mode`.

## Scorer versioning

Scorer versions pin both model identities exactly so scores are comparable across projects and time:

| Version | LM | Embedder |
|---------|----|----------|
| `1.0` | `Qwen/Qwen2.5-0.5B` | `sentence-transformers/all-MiniLM-L6-v2` |

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
