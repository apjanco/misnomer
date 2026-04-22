from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AlignmentType = Literal["MATCH", "SUBSTITUTION", "INSERTION", "DELETION"]
ErrorType = Literal["SEMANTIC", "OBVIOUS"]
ScorerMode = Literal["full", "standard", "text_only"]


class WordScore(BaseModel):
    predicted_word: str
    ground_truth_word: str
    alignment_type: AlignmentType
    error_type: ErrorType | None = None
    perplexity: float | None = None
    embedding_similarity: float = Field(default=1.0, ge=0.0, le=1.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    char_edit_distance: int = Field(default=0, ge=0)


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
    wer: float = Field(ge=0.0)
    metadata: dict[str, object] = Field(default_factory=dict)
