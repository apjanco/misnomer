# **Misnomer (Semantic Paraphasia)** Ã¢â‚¬â€ Semantic Error Measurement for OCR/HTR

### Package Specification v0.1

**Authors:** Andrew Janco, Wouter Haverals **Status:** Draft

---

## 1\. Problem Statement

Vision-Language Models (VLMs) used for OCR and HTR produce errors that are qualitatively different from traditional OCR failures. Rather than dropping characters or producing illegible noise, VLMs tend to substitute semantically plausible alternatives Ã¢â‚¬â€ synonyms, near-synonyms, contextually appropriate words Ã¢â‚¬â€ that pass human review undetected. These **semantic errors** are the most dangerous class of transcription mistake precisely because they are readable and feel correct.

Existing metrics (CER, WER) treat all substitutions equally. A character-level edit distance of 5 applies the same weight to `horse Ã¢â€ â€™ honse` (obvious, detectable) and `horse Ã¢â€ â€™ steed` (semantic, undetectable). This package is designed to make the second class of error visible, measurable, and legible to users.

---

## 2\. Goals

- Provide a **model-agnostic, reproducible** measure of semantic error in OCR/HTR output  
- Work regardless of what system produced the transcription (local VLM, remote API, Tesseract, Transkribus, etc.)  
- Deliver a **per-word and document-level score** that can drive highlighting in a UI  
- Distinguish between **semantic errors** (plausible substitutions) and **obvious errors** (clearly wrong output)  
- Use a **fixed, versioned internal scorer** so scores are comparable across projects and institutions  
- Degrade gracefully when logprob information is unavailable

---

## 3\. Core Concepts

### 3.1 The Separation of Concerns

The package decouples **transcription** from **evaluation**. Whatever system produced the predicted text is irrelevant. `misnomer` treats both predicted and ground truth text as plain strings and applies its own fixed internal models to score them.

\[Any OCR/HTR system\] Ã¢â€ â€™ predicted\_text

                                        \\

                                         Ã¢â€ â€™ misnomer.score() Ã¢â€ â€™ SemanticErrorReport

                                        /

              ground\_truth\_text \--------

### 3.2 Error Taxonomy

| Type | Description | Example |
| :---- | :---- | :---- |
| `SEMANTIC` | Substitution that is semantically close to ground truth; likely to pass human review | `horse Ã¢â€ â€™ steed` |
| `OBVIOUS` | Substitution that is semantically distant; likely detectable on reading | `horse Ã¢â€ â€™ house` |
| `INSERTION` | Word present in prediction, absent in ground truth | Ã¢â‚¬â€ |
| `DELETION` | Word present in ground truth, absent in prediction | Ã¢â‚¬â€ |
| `MATCH` | Correct transcription | Ã¢â‚¬â€ |

Only `SEMANTIC` and `OBVIOUS` substitutions receive a composite error score. The distinction is driven by embedding similarity between predicted and ground truth words.

### 3.3 The Two Signals

**Signal 1 Ã¢â‚¬â€ Perplexity of ground truth in context** Using a fixed internal language model, compute how surprised the model is by each ground truth word given its surrounding context. High perplexity means the correct word was contextually unlikely Ã¢â‚¬â€ the conditions under which semantic substitution is most likely to occur and hardest to catch.

**Signal 2 Ã¢â‚¬â€ Embedding similarity between predicted and ground truth** Using a fixed internal sentence-transformer, compute the semantic distance between the predicted word and the ground truth word. High similarity \= semantic error. Low similarity \= obvious error.

These two signals combine into a composite score per substituted word pair.

---

## 4\. Scoring Model Selection

The scoring models are fixed within a package version to ensure reproducibility. They are chosen to be small, open-licensed, and runnable on CPU for accessibility.

### 4.1 Language Model for Perplexity Scoring

The LM scorer does not need to be the same model that produced the transcription. Its job is to assign a probability to each ground truth token in context, acting as an independent linguistic auditor.

**Recommended default: `Qwen2.5-0.5B`**

Rationale: Qwen2.5 is trained on an 18-trillion-token multilingual corpus, giving it strong coverage across modern and historical language varieties including non-Latin scripts. The 0.5B variant runs on CPU with modest RAM. Critically, because Nanonets-OCR-s, Nanonets-OCR2, and Chandra are all fine-tuned on top of Qwen2.5-VL-3B, using a Qwen2.5 base LM as the perplexity scorer creates a linguistically coherent evaluation Ã¢â‚¬â€ the scorer shares the same underlying language model family as the most common VLM transcription systems, without being the same model.

**Alternative: `GPT-2` (124M)** Well-understood, runs anywhere, no license restrictions. Weaker on non-English and historical text. Useful as a fast fallback or for English-only collections.

**Future / domain-specific option: fine-tuned Qwen2.5-0.5B on historical corpora** For archival HTR work, a version fine-tuned on 18thÃ¢â‚¬â€œ19th century text would give meaningfully better perplexity signal. This is a future roadmap item (see Section 9).

### 4.2 Embedding Model for Semantic Distance

**Recommended default: `sentence-transformers/all-MiniLM-L6-v2`**

Rationale: Small (80MB), fast on CPU, strong semantic similarity performance for English. For multilingual collections, `paraphrase-multilingual-MiniLM-L12-v2` is a drop-in alternative.

---

## 5\. API Design

### 5.1 Primary Interface

import misnomer

report \= misnomer.score(

    predicted="The horse stood in the field near the chapel.",

    ground\_truth="The house stood in the field near the chapel.",

    scorer\_version="1.0"  \# explicit for reproducibility

)

print(report.document\_score)       \# float, 0.0Ã¢â‚¬â€œ1.0

print(report.word\_scores)          \# list of WordScore objects

print(report.semantic\_error\_count) \# int

print(report.obvious\_error\_count)  \# int

### 5.2 WordScore Object

@dataclass

class WordScore:

    predicted\_word: str

    ground\_truth\_word: str

    alignment\_type: str          \# MATCH | SUBSTITUTION | INSERTION | DELETION

    error\_type: str | None       \# SEMANTIC | OBVIOUS | None

    perplexity: float | None     \# perplexity of GT word in context

    embedding\_similarity: float  \# cosine similarity, predicted vs GT

    composite\_score: float       \# 0.0 (no error) to 1.0 (severe error)

    char\_edit\_distance: int      \# normalized Levenshtein

### 5.3 SemanticErrorReport Object

@dataclass

class SemanticErrorReport:

    predicted\_text: str

    ground\_truth\_text: str

    scorer\_version: str

    lm\_model: str                  \# name of LM used for perplexity

    embedder\_model: str            \# name of embedder used

    word\_scores: list\[WordScore\]

    document\_score: float          \# document-level aggregate

    semantic\_error\_count: int

    obvious\_error\_count: int

    wer: float                     \# standard WER for comparison

    metadata: dict                 \# optional, user-supplied

### 5.4 Batch Scoring

\# Score multiple document pairs

reports \= misnomer.score\_batch(

    pairs=\[("predicted\_1", "gt\_1"), ("predicted\_2", "gt\_2")\],

    scorer\_version="1.0"

)

### 5.5 Highlighting Helper

A utility to produce HTML with highlighted errors for display in a viewer:

html \= misnomer.highlight(

    report,

    semantic\_color="\#FF6B6B",  \# red for semantic errors

    obvious\_color="\#FFD93D",   \# yellow for obvious errors

    threshold=0.3              \# minimum composite score to highlight

)

---

## 6\. Internal Architecture

misnomer/

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ \_\_init\_\_.py              \# public API: score(), score\_batch(), highlight()

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ aligner.py               \# word-level sequence alignment (WER-style)

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ scorer.py                \# perplexity \+ embedding distance computation

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ classifier.py            \# SEMANTIC vs OBVIOUS classification logic

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ composite.py             \# combining signals into composite score

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ highlight.py             \# HTML highlighting utility

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ report.py                \# SemanticErrorReport \+ WordScore dataclasses

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ models/

Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ lm.py                \# LM loader \+ perplexity computation

Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ embedder.py          \# sentence-transformer loader \+ similarity

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ cli.py                   \# command-line interface

Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ tests/

    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ test\_aligner.py

    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ test\_scorer.py

    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ test\_classifier.py

    Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ fixtures/            \# sample predicted/GT pairs for testing

### 6.1 Alignment (`aligner.py`)

Uses dynamic programming (Needleman-Wunsch / WER-style alignment) to produce word-level aligned pairs before scoring. This is the same algorithm used to compute standard WER.

predicted:    "The horse stood in the field"

ground truth: "The house stood in the field"

              Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

alignment:    MATCH  SUBST  MATCH  MATCH MATCH MATCH

Alignment output is a list of `(predicted_word, ground_truth_word, alignment_type)` tuples passed to the scorer.

### 6.2 Perplexity Computation (`models/lm.py`)

For each substituted word pair, the ground truth word's perplexity is computed in the context of the surrounding ground truth sentence. This requires a single forward pass over the ground truth text with the internal LM.

Token-to-word aggregation: average logprob across tokens belonging to a word, normalized by character length of the word (not token count) to avoid penalizing long words.

Perplexity per word: `exp(-avg_logprob_per_char)`

### 6.3 Semantic Distance (`models/embedder.py`)

Word-level embeddings are computed for both predicted and ground truth words using the internal sentence-transformer. Cosine similarity is computed between the two embeddings.

`embedding_similarity = cosine_similarity(embed(predicted_word), embed(ground_truth_word))`

High similarity (close to 1.0) \= semantic error candidate. Low similarity (close to 0.0) \= obvious error candidate.

**Threshold for SEMANTIC vs OBVIOUS:** `embedding_similarity Ã¢â€°Â¥ 0.5` Ã¢â€ â€™ SEMANTIC. Below Ã¢â€ â€™ OBVIOUS. This threshold is configurable.

### 6.4 Composite Score (`composite.py`)

The composite score per substituted word pair combines perplexity and embedding similarity:

composite \= w1 \* normalized\_perplexity \+ w2 \* (1 \- embedding\_similarity)

Where:

- `normalized_perplexity` is the GT word's perplexity, clipped and normalized to \[0, 1\] relative to the document's perplexity range  
- `1 - embedding_similarity` is the semantic error contribution (high similarity \= high error)  
- Default weights: `w1 = 0.4`, `w2 = 0.6` (configurable)

Note: a high composite score indicates a concerning substitution. A semantically close substitution that surprised the LM (high perplexity, high embedding similarity) scores highest.

### 6.5 Document-Level Score

Weighted average of word-level composite scores across all substituted pairs, weighted by word frequency rank (rare words weighted higher, as errors on rare words are harder to catch).

---

## 7\. Tiered Logprob Support

The package supports three operational tiers depending on what information is available from the transcription system. The perplexity signal always comes from the **internal fixed LM**, so this tier distinction affects only whether additional confidence data from the OCR model itself can be incorporated.

| Tier | Condition | Behavior |
| :---- | :---- | :---- |
| **Full** | Local model, full logprobs available | Internal LM perplexity \+ optional OCR model confidence weighting |
| **Standard** | Remote API, predicted logprobs only | Internal LM perplexity only (OCR confidence not used) |
| **Text-only** | No logprobs available | Embedding distance \+ char edit distance only; no perplexity |

Text-only mode is weaker but still classifies SEMANTIC vs OBVIOUS errors and produces a meaningful composite score.

---

## 8\. Versioning and Reproducibility

Scorer versions pin both the LM and embedder model exactly:

| Version | LM | Embedder |
| :---- | :---- | :---- |
| `1.0` | `Qwen2.5-0.5B` | `all-MiniLM-L6-v2` |

Scores computed with different versions are **not comparable**. The `scorer_version` field in every report makes this explicit. If the bundled models change, the major version increments and a migration note is published.

Users can specify a custom LM or embedder for experimental use, but doing so disables the version guarantee and is flagged in the report:

report \= misnomer.score(

    predicted=...,

    ground\_truth=...,

    lm\_model="gpt2",          \# custom Ã¢â‚¬â€ disables reproducibility guarantee

    scorer\_version="custom"

)

---

## 9\. CLI

\# Score a single pair

misnomer score \--predicted pred.txt \--ground-truth gt.txt \--output report.json

\# Batch scoring from a CSV (columns: predicted, ground\_truth)

misnomer score-batch \--input pairs.csv \--output reports.json

\# Generate HTML highlighting

misnomer highlight \--report report.json \--output highlighted.html

\# Show package and scorer version

misnomer version

---

## 10\. Installation and Dependencies

pip install misnomer

Core dependencies:

- `transformers` Ã¢â‚¬â€ LM loading and forward pass  
- `sentence-transformers` Ã¢â‚¬â€ embedding model  
- `torch` Ã¢â‚¬â€ tensor operations (CPU sufficient for default models)  
- `numpy` Ã¢â‚¬â€ alignment and scoring math  
- `dataclasses` (stdlib)

Optional:

- `rich` Ã¢â‚¬â€ CLI output formatting

Model weights are downloaded on first use from HuggingFace Hub and cached locally. Total download: \~500MB for default models.

---

## 11\. Roadmap

| Priority | Item |
| :---- | :---- |
| High | Core implementation: aligner, scorer, classifier, report |
| High | Test suite with known semantic error fixtures |
| Medium | HTML highlight utility |
| Medium | CLI |
| Medium | Batch scoring with progress reporting |
| Low | Domain-adapted LM variant for historical/archival text (fine-tuned Qwen2.5-0.5B on 18thÃ¢â‚¬â€œ19th century corpora) |
| Low | Multilingual embedder as default for non-English collections |
| Low | Integration examples for Nanonets-OCR2, Chandra, Tesseract, Transkribus |

---

## 12\. Related Work and Context

- **Nanonets-OCR-s / OCR2**: 3.75B parameter VLMs fine-tuned from Qwen2.5-VL-3B for image-to-markdown OCR. Primary motivating system for this package.  
- **Chandra / Chandra-OCR-2**: OCR VLM from Datalab with strong multilingual support (90+ languages) and handwriting recognition; also built on Qwen2.5-VL architecture.  
- **WER / CER**: Standard metrics that treat all substitutions equally; `misnomer` is designed to complement, not replace, these metrics.  
- **lmppl**: Existing Python library for perplexity scoring across LM types (causal, masked, encoder-decoder). May be used internally or as a reference implementation.


