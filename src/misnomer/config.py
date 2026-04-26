from __future__ import annotations

from pydantic import BaseModel, Field


class ScorerConfig(BaseModel):
    scorer_version: str = "1.0"
    lm_model: str = "Qwen/Qwen2.5-0.5B"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    multilingual_embedder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    semantic_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    perplexity_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    local_files_only: bool = True
    allow_download: bool = False
    model_revision: str | None = None
    use_multilingual_embedder: bool = False

    def resolved_embedder_model(self) -> str:
        if self.use_multilingual_embedder:
            return self.multilingual_embedder_model
        return self.embedder_model

    def model_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "local_files_only": self.local_files_only and not self.allow_download,
        }
        if self.model_revision:
            kwargs["revision"] = self.model_revision
        return kwargs
