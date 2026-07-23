from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from rapidfuzz.distance import Levenshtein

from misnomer.aligner import align_words, char_edit_distance, tokenize_words
from misnomer.anchor import WordAnchors
from misnomer.classifier import SpellDictionary, classify_substitution, classify_substitution_v2
from misnomer.traditions import resolve_tradition
from misnomer.composite import composite_score, contrast_to_override_risk, normalize_perplexities
from misnomer.config import ScorerConfig
from misnomer.models.embedder import Embedder
from misnomer.models.lm import LMScorer
from misnomer.preprocess import preprocess
from misnomer.report import SemanticErrorReport, ScorerMode, WordScore, DocumentErrorType


class StrictModeError(RuntimeError):
    """Raised when strict=True but the full scorer stack is unavailable."""


def _enforce_strict(
    cfg: ScorerConfig,
    lm: LMScorer,
    embedder: Embedder,
    dictionary: SpellDictionary | None = None,
) -> None:
    if not cfg.strict:
        return
    missing: list[str] = []
    if not lm.is_transformer_backed:
        missing.append(f"LM {cfg.lm_model!r} (frequency-proxy fallback active)")
    if not embedder.is_model_backed:
        missing.append(
            f"embedder {cfg.resolved_embedder_model()!r} (char-similarity fallback active)"
        )
    if cfg.gates_active and (dictionary is None or not dictionary.available):
        missing.append(
            f"dictionary {cfg.dictionary_lang!r} (GARBLE gate off; install misnomer[gates])"
        )
    if missing:
        raise StrictModeError(
            "strict mode requires the full scorer stack; unavailable: "
            + "; ".join(missing)
            + ". Install `misnomer[model]`, pre-download the models, or set allow_download=True."
        )


def _frequency_weight(word: str, counts: dict[str, int]) -> float:
    freq = counts.get(word, 1)
    return 1.0 / float(freq)


def _build_dictionary(cfg: ScorerConfig) -> SpellDictionary | None:
    return SpellDictionary(cfg.dictionary_lang) if cfg.gates_active else None


def _score_with_models(
    predicted: str | None,
    ground_truth: str | None,
    lm: LMScorer,
    embedder: Embedder,
    cfg: ScorerConfig,
    metadata: dict[str, object] | None = None,
    dictionary: SpellDictionary | None = None,
) -> SemanticErrorReport:
    """Core scoring logic. Accepts pre-built model instances so callers can
    amortize the model-loading cost across many document pairs."""
    _enforce_strict(cfg, lm, embedder, dictionary)
    ground_truth = ground_truth or ""

    profile = resolve_tradition(cfg) if cfg.gates_active else None

    pre = preprocess(predicted)
    predicted = pre.text

    if pre.is_refusal:
        refusal_cer = Levenshtein.distance(predicted, ground_truth) / max(1, len(ground_truth))
        return SemanticErrorReport(
            predicted_text=predicted,
            ground_truth_text=ground_truth,
            scorer_version=cfg.scorer_version,
            lm_model=lm.resolved_model_name,
            embedder_model=embedder.resolved_model_name,
            scorer_mode="text_only",
            word_scores=[],
            document_score=1.0,
            semantic_error_count=0,
            obvious_error_count=0,
            wer=1.0,
            cer=refusal_cer,
            insertion_ratio=0.0,
            substitution_rate=1.0,
            document_embedding_similarity=None,
            document_error_type="refusal",
            is_refusal=True,
            strict=cfg.strict,
            tradition=profile.name if profile else None,
            preprocessing_applied=pre.transformations,
            metadata=metadata or {},
        )

    alignment = align_words(predicted, ground_truth, tokenizer=lm.tokenizer)

    gt_words = tokenize_words(ground_truth, tokenizer=lm.tokenizer)
    gt_counts: dict[str, int] = {}
    for w in gt_words:
        gt_counts[w] = gt_counts.get(w, 0) + 1

    # Determine operational tier based on available backends.
    if lm.is_transformer_backed and embedder.is_model_backed:
        scorer_mode: ScorerMode = "full"
    elif lm.is_transformer_backed:
        scorer_mode = "standard"
    else:
        scorer_mode = "text_only"

    gt_perplexities = lm.word_perplexities(gt_words)
    substitution_gt_indices: list[int] = []
    substitution_words: dict[int, str] = {}

    # v1.1: character anchoring over the raw strings, valid only when its
    # whitespace tokenization reproduces the alignment's token sequences
    # exactly (else anchor fields stay None — safe degradation).
    anchors: WordAnchors | None = None
    if cfg.gates_active:
        candidate = WordAnchors(predicted, ground_truth)
        align_pred = [it.predicted_word for it in alignment
                      if it.alignment_type in {"MATCH", "SUBSTITUTION", "INSERTION"}]
        align_gt = [it.ground_truth_word for it in alignment
                    if it.alignment_type in {"MATCH", "SUBSTITUTION", "DELETION"}]
        if candidate.pred_texts() == align_pred and candidate.gt_texts() == align_gt:
            anchors = candidate

    # (anchor_confidence, segmentation) per substitution, in alignment order
    anchor_info: list[tuple[float | None, str | None]] = []

    gt_idx = -1
    pred_idx = -1
    for item in alignment:
        if item.alignment_type in {"MATCH", "SUBSTITUTION", "INSERTION"}:
            pred_idx += 1
        if item.alignment_type in {"MATCH", "SUBSTITUTION", "DELETION"}:
            gt_idx += 1
        if item.alignment_type == "SUBSTITUTION":
            substitution_gt_indices.append(gt_idx)
            if anchors is not None:
                confidence, segmentation = anchors.describe(pred_idx, gt_idx)
            else:
                confidence, segmentation = None, None
            anchor_info.append((confidence, segmentation))
            # segmentation events are not readings: no Δ is computed for them
            if segmentation is None:
                substitution_words[gt_idx] = item.predicted_word

    substitution_raw_perplexities = [gt_perplexities[i] for i in substitution_gt_indices]
    substitution_norm_perplexities = normalize_perplexities(substitution_raw_perplexities)

    # v1.1: Δ per substituted position, one batched call per document.
    deltas: dict[int, float | None] = {}
    if cfg.gates_active and substitution_words:
        deltas = lm.substitution_deltas(gt_words, substitution_words)

    word_scores: list[WordScore] = []
    class_counts: Counter[str] = Counter()
    substitutions = 0
    insertions = 0
    deletions = 0

    subst_cursor = 0
    weighted_sum = 0.0
    weight_total = 0.0

    for item in alignment:
        distance = char_edit_distance(item.predicted_word, item.ground_truth_word)

        if item.alignment_type == "MATCH":
            ws = WordScore(
                predicted_word=item.predicted_word,
                ground_truth_word=item.ground_truth_word,
                alignment_type="MATCH",
                embedding_similarity=1.0,
                composite_score=0.0,
                char_edit_distance=distance,
            )
            word_scores.append(ws)
            continue

        if item.alignment_type == "SUBSTITUTION":
            substitutions += 1
            position = substitution_gt_indices[subst_cursor]
            perplexity = substitution_raw_perplexities[subst_cursor]
            normalized_perplexity = substitution_norm_perplexities[subst_cursor]
            subst_cursor += 1

            # Surprisal contrast (scorer_version >= 1.1). Both candidates are
            # scored under the SAME gold left-context so the intrinsic rarity of
            # the slot cancels in the difference; Δ = S(pred) − S(gold) < 0 is
            # the language-prior-override signature. In eval mode the context is
            # the gold prefix (available); a live/inference caller would supply
            # the correctly-read predicted prefix instead.
            surprisal_gold: float | None = None
            surprisal_pred: float | None = None
            surprisal_contrast: float | None = None
            override_risk: float | None = None
            # gates_active keeps scorer_version="1.0" bit-identical: without
            # it the contrast fields populate (and the composite's perplexity
            # leg switches to override_risk) under a pinned 1.0 config.
            if cfg.gates_active and cfg.use_surprisal_contrast and lm.is_transformer_backed:
                left_context = " ".join(gt_words[:position])
                surprisal_gold = lm.first_token_surprisal(left_context, item.ground_truth_word)
                surprisal_pred = lm.first_token_surprisal(left_context, item.predicted_word)
                if surprisal_gold is not None and surprisal_pred is not None:
                    surprisal_contrast = surprisal_pred - surprisal_gold
                    override_risk = contrast_to_override_risk(
                        surprisal_contrast, cfg.contrast_temperature
                    )

            if embedder.is_model_backed:
                similarity: float | None = embedder.similarity(
                    item.predicted_word, item.ground_truth_word
                )
            else:
                similarity = None

            confidence, segmentation = (
                anchor_info[subst_cursor - 1] if cfg.gates_active else (None, None)
            )
            if segmentation is not None:
                error_type = "SEGMENTATION"
            elif cfg.gates_active:
                error_type = classify_substitution_v2(
                    item.predicted_word,
                    item.ground_truth_word,
                    similarity,
                    cfg,
                    dictionary,
                    profile=profile,
                )
            elif similarity is not None:
                error_type = classify_substitution(similarity, cfg.semantic_threshold)
            else:
                error_type = None

            # Perplexity leg of the composite: the signed-contrast override risk
            # when available (word not at position 0, contrast enabled), else
            # the gold-only normalized perplexity (scorer 1.0 behaviour).
            perplexity_leg = override_risk if override_risk is not None else normalized_perplexity
            score_value = composite_score(
                perplexity_leg,
                similarity,
                perplexity_weight=cfg.perplexity_weight,
                semantic_weight=cfg.semantic_weight,
            )

            if error_type is not None:
                class_counts[error_type] += 1

            wt = _frequency_weight(item.ground_truth_word, gt_counts)
            weighted_sum += score_value * wt
            weight_total += wt

            ws = WordScore(
                predicted_word=item.predicted_word,
                ground_truth_word=item.ground_truth_word,
                alignment_type="SUBSTITUTION",
                error_type=error_type,
                perplexity=perplexity,
                embedding_similarity=similarity,
                composite_score=score_value,
                char_edit_distance=distance,
                delta=deltas.get(position) if cfg.gates_active else None,
                context_words=position if cfg.gates_active else None,
                anchor_confidence=confidence,
                surprisal_gold=surprisal_gold,
                surprisal_pred=surprisal_pred,
                surprisal_contrast=surprisal_contrast,
                override_risk=override_risk,
            )
            word_scores.append(ws)
            continue

        if item.alignment_type == "INSERTION":
            insertions += 1
            word_scores.append(
                WordScore(
                    predicted_word=item.predicted_word,
                    ground_truth_word=item.ground_truth_word,
                    alignment_type="INSERTION",
                    embedding_similarity=0.0,
                    composite_score=0.0,
                    char_edit_distance=distance,
                )
            )
            continue

        deletions += 1
        word_scores.append(
            WordScore(
                predicted_word=item.predicted_word,
                ground_truth_word=item.ground_truth_word,
                alignment_type="DELETION",
                embedding_similarity=0.0,
                composite_score=0.0,
                char_edit_distance=distance,
            )
        )

    document_score = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    wer = (substitutions + insertions + deletions) / max(1, len(gt_words))

    gt_chars = len(ground_truth)
    cer = Levenshtein.distance(predicted, ground_truth) / max(1, gt_chars)

    pred_words = tokenize_words(predicted, tokenizer=lm.tokenizer)
    insertion_ratio = insertions / max(1, len(pred_words))
    substitution_rate = substitutions / max(1, len(gt_words))

    doc_sim = embedder.document_similarity(predicted, ground_truth)

    semantic_error_count = class_counts["SEMANTIC"]
    obvious_error_count = class_counts["OBVIOUS"]

    def _document_error_type() -> DocumentErrorType:
        if (
            doc_sim is not None
            and doc_sim < cfg.hallucination_similarity_threshold
            and substitution_rate >= cfg.hallucination_substitution_rate_threshold
        ):
            return "hallucinated"
        if semantic_error_count > 0 or obvious_error_count > 0 or wer > 0:
            return "partial"
        return "correct"

    return SemanticErrorReport(
        predicted_text=predicted,
        ground_truth_text=ground_truth,
        scorer_version=cfg.scorer_version,
        lm_model=lm.resolved_model_name,
        embedder_model=embedder.resolved_model_name,
        scorer_mode=scorer_mode,
        word_scores=word_scores,
        document_score=document_score,
        semantic_error_count=semantic_error_count,
        obvious_error_count=obvious_error_count,
        numeric_error_count=class_counts["NUMERIC"],
        normalization_count=class_counts["NORMALIZATION"],
        archaizing_count=class_counts["ARCHAIZING"],
        garble_count=class_counts["GARBLE"],
        homoglyph_count=class_counts["HOMOGLYPH"],
        segmentation_count=class_counts["SEGMENTATION"],
        wer=wer,
        cer=cer,
        insertion_ratio=insertion_ratio,
        substitution_rate=substitution_rate,
        document_embedding_similarity=doc_sim,
        document_error_type=_document_error_type(),
        is_refusal=False,
        strict=cfg.strict,
        tradition=profile.name if profile else None,
        preprocessing_applied=pre.transformations,
        metadata=metadata or {},
    )


def score(
    predicted: str | None,
    ground_truth: str | None,
    scorer_version: str = "1.0",
    metadata: dict[str, object] | None = None,
    config: ScorerConfig | None = None,
) -> SemanticErrorReport:
    cfg = config or ScorerConfig(scorer_version=scorer_version)
    lm = LMScorer(cfg)
    embedder = Embedder(cfg)
    dictionary = _build_dictionary(cfg)
    _enforce_strict(cfg, lm, embedder, dictionary)
    return _score_with_models(predicted, ground_truth, lm, embedder, cfg, metadata, dictionary)


def score_batch(
    pairs: list[tuple[str, str]] | list[tuple[str, str, dict]],
    scorer_version: str = "1.0",
    metadata: dict[str, object] | None = None,
    config: ScorerConfig | None = None,
) -> list[SemanticErrorReport]:
    """Score multiple document pairs, loading models only once."""
    cfg = config or ScorerConfig(scorer_version=scorer_version)
    lm = LMScorer(cfg)
    embedder = Embedder(cfg)
    dictionary = _build_dictionary(cfg)
    _enforce_strict(cfg, lm, embedder, dictionary)
    reports = []
    for pair in pairs:
        pred, gt = pair[0], pair[1]
        pair_meta: dict[str, object] = dict(pair[2]) if len(pair) > 2 else (metadata or {})  # type: ignore[misc]
        reports.append(_score_with_models(pred, gt, lm, embedder, cfg, pair_meta, dictionary))
    return reports


def score_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    predicted_field: str = "predicted",
    ground_truth_field: str = "ground_truth",
    scorer_version: str = "1.0",
    config: ScorerConfig | None = None,
) -> int:
    """Score document pairs from a JSONL file and write reports to another JSONL file.

    Streams both input and output so large files don't need to fit in memory.
    Models are loaded once for the entire file.

    Parameters
    ----------
    input_path:
        Path to a JSONL file. Each line must be a JSON object with at least
        ``predicted_field`` and ``ground_truth_field`` keys. All other fields
        are passed through into the report's ``metadata``.
    output_path:
        Path where scored reports are written as JSONL (one JSON object per line).
    predicted_field:
        Key in each input record that contains the predicted text.
    ground_truth_field:
        Key in each input record that contains the ground truth text.
    scorer_version:
        Scorer version string, pinned for reproducibility.
    config:
        Optional :class:`ScorerConfig` override.

    Returns
    -------
    int
        Number of records scored.
    """
    cfg = config or ScorerConfig(scorer_version=scorer_version)
    lm = LMScorer(cfg)
    embedder = Embedder(cfg)
    dictionary = _build_dictionary(cfg)
    _enforce_strict(cfg, lm, embedder, dictionary)

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with (
        input_path.open(encoding="utf-8") as in_fh,
        output_path.open("w", encoding="utf-8") as out_fh,
    ):
        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predicted = record.get(predicted_field) or ""
            ground_truth = record.get(ground_truth_field) or ""
            meta = {k: v for k, v in record.items() if k not in {predicted_field, ground_truth_field}}
            report = _score_with_models(predicted, ground_truth, lm, embedder, cfg, meta, dictionary)
            out_fh.write(report.model_dump_json() + "\n")
            count += 1

    return count
