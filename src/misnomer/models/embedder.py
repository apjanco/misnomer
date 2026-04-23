from __future__ import annotations

import logging

from rapidfuzz.fuzz import ratio

from misnomer.config import ScorerConfig

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, config: ScorerConfig):
        self.config = config
        self.model_name = config.resolved_embedder_model()
        self._model = None

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name, **self.config.model_kwargs())
        except (ImportError, OSError, ValueError) as exc:
            log.warning("Embedder: could not load %r (%s: %s); falling back to rapidfuzz char-similarity.",
                        self.model_name, type(exc).__name__, exc)
            self._model = None

    @property
    def is_model_backed(self) -> bool:
        return self._model is not None

    @property
    def resolved_model_name(self) -> str:
        return self.model_name if self._model is not None else "rapidfuzz-char-similarity"

    def similarity(self, left: str, right: str) -> float:
        if not left and not right:
            return 1.0
        if self._model is None:
            return ratio(left, right) / 100.0

        embeddings = self._model.encode([left, right], normalize_embeddings=True)
        score = float((embeddings[0] * embeddings[1]).sum())
        return max(0.0, min(1.0, score))

