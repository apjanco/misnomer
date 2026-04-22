from __future__ import annotations

from rapidfuzz.fuzz import ratio

from misnomer.config import ScorerConfig


class Embedder:
    def __init__(self, config: ScorerConfig):
        self.config = config
        self.model_name = config.resolved_embedder_model()
        self._model = None

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name, **self.config.model_kwargs())
        except Exception:
            self._model = None

    def similarity(self, left: str, right: str) -> float:
        if not left and not right:
            return 1.0
        if self._model is None:
            return ratio(left, right) / 100.0

        embeddings = self._model.encode([left, right], normalize_embeddings=True)
        score = float((embeddings[0] * embeddings[1]).sum())
        return max(0.0, min(1.0, score))

