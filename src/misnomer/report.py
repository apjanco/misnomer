from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AlignmentType = Literal["MATCH", "SUBSTITUTION", "INSERTION", "DELETION"]
ErrorType = Literal[
    "SEMANTIC",
    "OBVIOUS",
    # v1.1 gate classes (scorer_version >= 1.1):
    "NUMERIC",
    "NORMALIZATION",
    "ARCHAIZING",
    "GARBLE",
    "HOMOGLYPH",
    # v1.1: character anchoring says this "substitution" is a segmentation
    # event (join/split/fragment), not one word read as another.
    "SEGMENTATION",
]
ScorerMode = Literal["full", "standard", "text_only"]
DocumentErrorType = Literal["correct", "partial", "hallucinated", "refusal"]


class WordScore(BaseModel):
    predicted_word: str
    ground_truth_word: str
    alignment_type: AlignmentType
    error_type: ErrorType | None = None
    perplexity: float | None = None
    embedding_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    char_edit_distance: int = Field(default=0, ge=0)
    # v1.1: Δ = log P(predicted|context) − log P(truth|context), full-word;
    # positive = the auditor's prior prefers what was written.
    delta: float | None = None
    # v1.1: ground-truth words preceding the substitution (Δ reliability flag;
    # 0–1 context words degenerate toward a frequency contrast).
    context_words: int | None = None
    # v1.1: fraction of identically anchored characters between the two
    # words under character-level alignment (see anchor.py); low values
    # mark pairs the aligner matched by position rather than by ink.
    anchor_confidence: float | None = None
    # Surprisal-contrast signal (scorer_version >= 1.1). Populated on
    # SUBSTITUTIONs when the transformer-backed LM is available and
    # use_surprisal_contrast is enabled. Both candidates are scored under
    # the same gold left-context, so surprisal_contrast == −delta.
    surprisal_gold: float | None = None       # −log P(gold_word | context), nats
    surprisal_pred: float | None = None       # −log P(pred_word | context), nats
    surprisal_contrast: float | None = None    # Δ = surprisal_pred − surprisal_gold
    override_risk: float | None = Field(default=None, ge=0.0, le=1.0)  # σ(−Δ/T)


class SemanticErrorReport(BaseModel):
    predicted_text: str
    ground_truth_text: str
    scorer_version: str
    lm_model: str
    embedder_model: str
    scorer_mode: ScorerMode = "text_only"
    word_scores: list[WordScore]
    document_score: float = Field(ge=0.0, le=1.0)
    semantic_error_count: int = Field(ge=0)
    obvious_error_count: int = Field(ge=0)
    # v1.1 gate-class counts (always 0 under scorer_version < 1.1):
    numeric_error_count: int = Field(default=0, ge=0)
    normalization_count: int = Field(default=0, ge=0)
    archaizing_count: int = Field(default=0, ge=0)
    garble_count: int = Field(default=0, ge=0)
    homoglyph_count: int = Field(default=0, ge=0)
    segmentation_count: int = Field(default=0, ge=0)
    # v1.1 provenance: which orthographic tradition profile the allograph
    # gate ran under (None under scorer_version < 1.1).
    tradition: str | None = None
    wer: float = Field(ge=0.0)
    cer: float = Field(ge=0.0)
    insertion_ratio: float = Field(default=0.0, ge=0.0)
    substitution_rate: float = Field(default=0.0, ge=0.0)
    document_embedding_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    document_error_type: DocumentErrorType = "correct"
    is_refusal: bool = False
    strict: bool = False
    preprocessing_applied: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
