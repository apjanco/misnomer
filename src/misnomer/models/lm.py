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

    _DTYPES = ("float32", "float16", "bfloat16")

    def __init__(self, config: ScorerConfig):
        self.config = config
        self.model_name = config.lm_model
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = "cpu"

        # Config errors must raise, never silently degrade to the proxy.
        if config.torch_dtype not in self._DTYPES:
            raise ValueError(
                f"torch_dtype must be one of {self._DTYPES}, got {config.torch_dtype!r}"
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            kwargs = self.config.model_kwargs()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=getattr(torch, config.torch_dtype),
                **kwargs,
            )
            self._model.eval()
            self._device = self._resolve_device(torch, config.device)
            self._model.to(self._device)
            self._torch = torch
        except (ImportError, OSError, ValueError) as exc:
            log.warning("LMScorer: could not load %r (%s: %s); falling back to frequency proxy.",
                        self.model_name, type(exc).__name__, exc)
            self._model = None
            self._tokenizer = None
            self._torch = None

    @staticmethod
    def _resolve_device(torch, device: str) -> str:
        if device != "auto":
            return device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def is_transformer_backed(self) -> bool:
        return self._model is not None

    @property
    def tokenizer(self):
        """The loaded HuggingFace tokenizer, or ``None`` when in proxy mode."""
        return self._tokenizer

    @property
    def resolved_model_name(self) -> str:
        return self.model_name if self._model is not None else "frequency-proxy"

    @property
    def resolved_device(self) -> str:
        return self._device if self._model is not None else "none"

    @property
    def resolved_dtype(self) -> str:
        return self.config.torch_dtype if self._model is not None else "none"

    def word_perplexities(self, ground_truth_words: list[str]) -> dict[int, float]:
        if not ground_truth_words:
            return {}
        if self._model is not None and self._tokenizer is not None:
            return self._transformer_perplexities(ground_truth_words)
        return self._proxy_perplexities(ground_truth_words)

    def first_token_surprisal(
        self, left_context: str, word: str
    ) -> float | None:
        """Surprisal (−log P) of *word*'s first token given *left_context*.

        This is the reusable primitive behind the surprisal *contrast* signal.
        Where :meth:`word_perplexities` scores every gold word in one pass and
        returns ``exp(−logprob)`` (perplexity), this scores a *single* candidate
        word placed after an arbitrary context and returns the raw natural-log
        surprisal ``−logprob`` — the additive, signed quantity that contrasts
        (``S(pred|c) − S(gold|c)``) are built from.

        Only the first token whose character span overlaps *word* is used, for
        the same reason as in :meth:`word_perplexities`: it is the token the LM
        predicts purely from context, whereas within-word continuation tokens
        are near-certain and would dilute the contextual signal.

        Parameters
        ----------
        left_context:
            The text preceding *word* (the correctly-read neighbours). May be
            empty, in which case *word* has no predecessor to condition on.
        word:
            The single candidate word to score.

        Returns
        -------
        float | None
            ``−logprob`` (nats, ≥ 0) of *word*'s first token, or ``None`` when
            the scorer is in frequency-proxy mode, *word* is empty, or the word
            lands at sequence position 0 (no predecessor token to condition on,
            e.g. empty *left_context*).
        """
        if self._model is None or self._tokenizer is None:
            return None
        if not word or not word.strip():
            return None

        import torch.nn.functional as F

        left = left_context or ""
        # A single space joins context and word so the word carries its natural
        # leading-space token (byte-level BPE encodes the space into the word's
        # first token). When there is no context the word starts the sequence.
        if left and not left.endswith((" ", "\n", "\t")):
            text = left + " " + word
        else:
            text = left + word
        word_start = len(text) - len(word)
        word_end = len(text)

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]
        if input_ids.shape[1] == 0:
            return None
        offset_mapping = enc.get("offset_mapping")

        with self._torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits[0]

        log_probs = F.log_softmax(logits, dim=-1)

        seq_len = input_ids.shape[1]
        if offset_mapping is None:
            return None
        offsets = offset_mapping[0].tolist()
        # Find the FIRST token whose character span overlaps the word.
        for tok_idx in range(seq_len):
            tok_start, tok_end = offsets[tok_idx]
            if tok_end > word_start and tok_start < word_end:
                # token i's logprob is predicted from position i-1; a word's
                # first token at sequence position 0 has no predecessor to
                # condition on, so it carries no contextual surprisal. Scoring
                # a later (within-word continuation) token instead would return
                # a near-certain character continuation, not a contextual
                # prediction — so return None rather than a misleading value.
                if tok_idx == 0:
                    return None
                lp = log_probs[tok_idx - 1, input_ids[0, tok_idx]].item()
                return -lp
        # No token overlaps the word span.
        return None

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

        # Guard: a model with seq_len==0 (empty text) cannot be forward-passed.
        if input_ids.shape[1] == 0:
            return {i: 1.0 for i in range(len(ground_truth_words))}
        # offset_mapping may be absent for slow tokenizers.
        offset_mapping = enc.get("offset_mapping")

        with self._torch.no_grad():
            outputs = self._model(input_ids.to(self._device))
            # float() for numerical stability under float16/bfloat16
            # (identity under the default float32).
            logits = outputs.logits[0].float().cpu()  # (seq_len, vocab_size)

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
                first_lp: float | None = None
                for tok_idx in range(1, seq_len):
                    tok_start, tok_end = offsets[tok_idx]
                    # Use only the first token whose span overlaps the word.
                    # Subsequent tokens are within-word continuations conditioned
                    # on the word already being started — near-certain predictions
                    # that dilute the contextual surprisal signal when averaged.
                    if tok_end > w_start and tok_start < w_end:
                        lp = token_log_probs[tok_idx]
                        if lp is not None:
                            first_lp = lp
                        break  # only the first overlapping token
                result[word_idx] = self._perplexity_from_logprob(first_lp)
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
    def _perplexity_from_logprob(logprob: float | None) -> float:
        """Convert a single token log-probability to perplexity.

        Only the first token of a word's span is used — it is the one the LM
        predicts purely from context.  Within-word continuation tokens are
        conditioned on the word already being started and inflate the average
        log-prob toward zero, deflating perplexity for multi-token words.
        """
        if logprob is None:
            return 1.0
        return math.exp(-logprob)

    # ------------------------------------------------------------------
    # Δ — the prior's preference for the substituted reading
    # ------------------------------------------------------------------

    def substitution_surprisals(
        self,
        ground_truth_words: list[str],
        substitutions: dict[int, str],
        chunk_size: int = 16,
    ) -> dict[int, tuple[float, float] | None]:
        """Full-word surprisals ``(s_gold, s_pred)`` per substituted position.

        ``s(w) = −log P(w | c_i)`` in nats, where c_i is the ground-truth
        prefix (words before position i) and the probability is the SUM over
        ALL of the word's tokens, teacher-forced on the same prefix —
        full-word scoring, never first-token-only. First-token contrasts
        fail two ways: orthographic neighbors frequently share their opening
        subword (``in|sure`` / ``in|jure`` — the contrast collapses to
        exactly zero), and when tokenizations differ in granularity the
        shorter first token carries the probability mass of every possible
        continuation, so fragment-like openings look spuriously fluent.

        Position 0 (empty prefix) is anchored on the tokenizer's BOS token,
        falling back to EOS — the contrast then degenerates gracefully
        toward a frequency contrast. Returns ``None`` per position in proxy
        mode or for empty words. :meth:`substitution_deltas` and the
        scorer's ``surprisal_contrast``/``override_risk`` fields all derive
        from this one batched computation.
        """
        if not substitutions:
            return {}
        if self._model is None or self._tokenizer is None:
            return {i: None for i in substitutions}

        torch = self._torch
        import torch.nn.functional as F

        tok = self._tokenizer
        anchor = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

        # Build one scoring entry per (position, candidate) pair.
        # entry = (position, role, prefix_len_tokens, full_ids)
        entries: list[tuple[int, str, int, list[int]]] = []
        results: dict[int, tuple[float, float] | None] = {}
        for i, predicted in substitutions.items():
            truth = ground_truth_words[i]
            if not predicted.strip() or not truth.strip():
                results[i] = None
                continue
            prefix = " ".join(ground_truth_words[:i])
            # Equal words (defensive; the aligner emits them as MATCH): score
            # once and reuse for both roles, so the contrast is exactly 0.0.
            roles = (
                (("pred", predicted), ("truth", truth))
                if predicted != truth
                else (("truth", truth),)
            )
            for role, word in roles:
                if prefix:
                    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
                    f_ids = tok(prefix + " " + word, add_special_tokens=False)["input_ids"]
                    # Prefix stability guard: BPE pre-tokenization keeps merges
                    # inside word boundaries, but verify rather than assume.
                    k = 0
                    while k < min(len(p_ids), len(f_ids)) and p_ids[k] == f_ids[k]:
                        k += 1
                    p_len = k
                else:
                    p_ids = [anchor] if anchor is not None else []
                    f_ids = p_ids + tok(word, add_special_tokens=False)["input_ids"]
                    p_len = len(p_ids)
                entries.append((i, role, p_len, f_ids))

        # Teacher-forced scoring, chunked; right padding is safe for causal
        # models (padded positions are never read for real-token logprobs).
        logprob_sums: dict[tuple[int, str], float] = {}
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else (anchor or 0)
        for start in range(0, len(entries), chunk_size):
            chunk = entries[start : start + chunk_size]
            if not chunk:
                continue
            max_len = max(len(e[3]) for e in chunk)
            batch = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
            for row, (_, _, _, f_ids) in enumerate(chunk):
                batch[row, : len(f_ids)] = torch.tensor(f_ids, dtype=torch.long)
                mask[row, : len(f_ids)] = 1
            with torch.no_grad():
                out = self._model(batch.to(self._device), attention_mask=mask.to(self._device))
                logits = out.logits.float().cpu()  # (rows, max_len, vocab)
            log_probs = F.log_softmax(logits, dim=-1)
            for row, (i, role, p_len, f_ids) in enumerate(chunk):
                total = 0.0
                # token at index j is predicted from logits at j-1; j >= 1
                for j in range(max(p_len, 1), len(f_ids)):
                    total += log_probs[row, j - 1, f_ids[j]].item()
                logprob_sums[(i, role)] = total

        for i in substitutions:
            if i in results:  # degenerate cases handled above
                continue
            s_gold = -logprob_sums[(i, "truth")]
            s_pred = -logprob_sums[(i, "pred")] if (i, "pred") in logprob_sums else s_gold
            results[i] = (s_gold, s_pred)
        return results

    def substitution_deltas(
        self,
        ground_truth_words: list[str],
        substitutions: dict[int, str],
        chunk_size: int = 16,
    ) -> dict[int, float | None]:
        """Δ_i = log P(ŵ_i | c_i) − log P(w*_i | c_i) per substituted position.

        Positive Δ: the model's prior prefers what was written over what the
        reference attests (lectio facilior as a sign test). Derived from
        :meth:`substitution_surprisals` as ``s_gold − s_pred``; see there for
        the full-word scoring rationale and position-0 anchoring.
        """
        pairs = self.substitution_surprisals(ground_truth_words, substitutions, chunk_size)
        return {i: (None if pair is None else pair[0] - pair[1]) for i, pair in pairs.items()}

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

