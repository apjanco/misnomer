from __future__ import annotations

import logging
import math
from collections import Counter

from misnomer.config import ScorerConfig

log = logging.getLogger(__name__)


class LMScorer:
    """Compute per-word perplexity using a causal language model.

    Falls back to a deterministic frequency-based proxy when ``transformers``
    or ``torch`` are not installed or the model is unavailable locally.
    """

    def __init__(self, config: ScorerConfig):
        self.config = config
        self.model_name = config.lm_model
        self._model = None
        self._tokenizer = None
        self._torch = None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            kwargs = self.config.model_kwargs()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                **kwargs,
            )
            self._model.eval()
            self._torch = torch
        except (ImportError, OSError, ValueError) as exc:
            log.warning("LMScorer: could not load %r (%s: %s); falling back to frequency proxy.",
                        self.model_name, type(exc).__name__, exc)
            self._model = None
            self._tokenizer = None
            self._torch = None

    @property
    def is_transformer_backed(self) -> bool:
        return self._model is not None

    @property
    def resolved_model_name(self) -> str:
        return self.model_name if self._model is not None else "frequency-proxy"

    def word_perplexities(self, ground_truth_words: list[str]) -> dict[int, float]:
        if self._model is not None and self._tokenizer is not None:
            return self._transformer_perplexities(ground_truth_words)
        return self._proxy_perplexities(ground_truth_words)

    # ------------------------------------------------------------------
    # Transformer path
    # ------------------------------------------------------------------

    def _transformer_perplexities(self, ground_truth_words: list[str]) -> dict[int, float]:
        import torch.nn.functional as F

        text = " ".join(ground_truth_words)

        # Tokenize with character-offset information (fast tokenizers only).
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]  # (1, seq_len)
        # offset_mapping may be absent for slow tokenizers.
        offset_mapping = enc.get("offset_mapping")

        with self._torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

        log_probs = F.log_softmax(logits, dim=-1)

        # For a causal LM, token i's logprob = log_probs[i-1][token_id[i]].
        # Position 0 (usually BOS) has no predecessor; we skip it.
        seq_len = input_ids.shape[1]
        token_log_probs: list[float | None] = [None]  # index 0 has no logprob
        for i in range(1, seq_len):
            token_log_probs.append(log_probs[i - 1, input_ids[0, i]].item())

        word_spans = self._word_char_spans(text, ground_truth_words)
        result: dict[int, float] = {}

        if offset_mapping is not None:
            offsets = offset_mapping[0].tolist()  # list of [start, end]
            for word_idx, (w_start, w_end) in enumerate(word_spans):
                gathered: list[float] = []
                for tok_idx in range(1, seq_len):
                    tok_start, tok_end = offsets[tok_idx]
                    # Include token if it overlaps with the word span.
                    if tok_end > w_start and tok_start < w_end:
                        lp = token_log_probs[tok_idx]
                        if lp is not None:
                            gathered.append(lp)
                word = ground_truth_words[word_idx]
                result[word_idx] = self._perplexity_from_logprobs(word, gathered)
        else:
            # Slow tokenizer fallback: distribute logprobs equally across words.
            valid_lps = [lp for lp in token_log_probs if lp is not None]
            avg_lp = sum(valid_lps) / max(1, len(valid_lps))
            for word_idx, word in enumerate(ground_truth_words):
                result[word_idx] = math.exp(-avg_lp)

        return result

    @staticmethod
    def _word_char_spans(text: str, words: list[str]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            if start == -1:
                start = pos
            end = start + len(word)
            spans.append((start, end))
            pos = end
        return spans

    @staticmethod
    def _perplexity_from_logprobs(word: str, logprobs: list[float]) -> float:
        if not logprobs:
            return 1.0
        avg_logprob = sum(logprobs) / len(logprobs)
        return math.exp(-avg_logprob)

    # ------------------------------------------------------------------
    # Deterministic proxy (used when transformers/torch unavailable)
    # ------------------------------------------------------------------

    def _proxy_perplexities(self, ground_truth_words: list[str]) -> dict[int, float]:
        counts = Counter(ground_truth_words)
        total = max(1, len(ground_truth_words))
        result: dict[int, float] = {}
        for idx, word in enumerate(ground_truth_words):
            freq = counts[word]
            rarity = math.log1p(total / max(1, freq))
            result[idx] = rarity
        return result

