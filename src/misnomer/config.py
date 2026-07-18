from __future__ import annotations

from pydantic import BaseModel, Field


class ScorerConfig(BaseModel):
    scorer_version: str = "1.0"
    lm_model: str = "Qwen/Qwen2.5-0.5B"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # --- v1.1 additions (behavior gated on scorer_version, see gates_active) ---
    strict: bool = False
    """Refuse to score unless the full model stack is available.

    The silent fallbacks (frequency-proxy LM, character-similarity embedder)
    invert the SEMANTIC/OBVIOUS taxonomy; strict mode turns that degradation
    into a hard error. All paper/production runs should set strict=True."""
    torch_dtype: str = "float32"  # "float32" | "float16" | "bfloat16"
    device: str = "cpu"  # "cpu" (default: bit-stable v1.0 behavior) | "auto" | "mps" | "cuda"
    numeric_gate: bool = True  # classify numeral-bearing substitutions as NUMERIC (v1.1 gates)
    dictionary_lang: str = "en"  # pyspellchecker language for the GARBLE gate (v1.1)
    tradition: str | dict | None = None
    """Orthographic tradition profile for the allograph gate (v1.1).

    A built-in name ("modern", "early-modern-latin-print"), an inline
    profile dict, or None for the default: "modern", i.e. no period
    equivalences unless explicitly declared — the classifier never
    applies a transcription convention silently."""
    allograph_map: dict[str, str] | None = None  # legacy: becomes a custom tradition profile
    multilingual_embedder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    semantic_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    perplexity_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    # Thresholds for document-level error type classification
    hallucination_similarity_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    hallucination_substitution_rate_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    local_files_only: bool = True
    allow_download: bool = False
    model_revision: str | None = None
    use_multilingual_embedder: bool = False

    def version_tuple(self) -> tuple[int, ...]:
        """Parse scorer_version into a comparable tuple; non-numeric parts
        contribute their digits ("1.1-rc" -> (1, 1)), fully non-numeric
        versions ("custom") compare as (0,)."""
        parts: list[int] = []
        for piece in self.scorer_version.split("."):
            digits = "".join(ch for ch in piece if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts) or (0,)

    @property
    def gates_active(self) -> bool:
        """Substitution gates (NUMERIC / allograph / GARBLE) and other
        v1.1 behavior activate from scorer_version >= 1.1."""
        return self.version_tuple() >= (1, 1)

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
