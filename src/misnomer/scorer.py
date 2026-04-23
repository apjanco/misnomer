from __future__ import annotations

from misnomer.aligner import align_words, char_edit_distance, tokenize_words
from misnomer.classifier import classify_substitution
from misnomer.composite import composite_score, normalize_perplexities
from misnomer.config import ScorerConfig
from misnomer.models.embedder import Embedder
from misnomer.models.lm import LMScorer
from misnomer.report import SemanticErrorReport, ScorerMode, WordScore


def _frequency_weight(word: str, counts: dict[str, int]) -> float:
    freq = counts.get(word, 1)
    return 1.0 / float(freq)


def score(
    predicted: str,
    ground_truth: str,
    scorer_version: str = "1.0",
    metadata: dict[str, object] | None = None,
    config: ScorerConfig | None = None,
) -> SemanticErrorReport:
    cfg = config or ScorerConfig(scorer_version=scorer_version)
    alignment = align_words(predicted, ground_truth)

    gt_words = tokenize_words(ground_truth)
    gt_counts: dict[str, int] = {}
    for w in gt_words:
        gt_counts[w] = gt_counts.get(w, 0) + 1

    lm = LMScorer(cfg)
    embedder = Embedder(cfg)

    # Determine operational tier based on available backends.
    if lm.is_transformer_backed and embedder.is_model_backed:
        scorer_mode: ScorerMode = "full"
    elif lm.is_transformer_backed:
        scorer_mode = "standard"
    else:
        scorer_mode = "text_only"

    gt_perplexities = lm.word_perplexities(gt_words)
    substitution_gt_indices: list[int] = []

    gt_idx = -1
    for item in alignment:
        if item.alignment_type in {"MATCH", "SUBSTITUTION", "DELETION"}:
            gt_idx += 1
        if item.alignment_type == "SUBSTITUTION":
            substitution_gt_indices.append(gt_idx)

    substitution_raw_perplexities = [gt_perplexities[i] for i in substitution_gt_indices]
    substitution_norm_perplexities = normalize_perplexities(substitution_raw_perplexities)

    word_scores: list[WordScore] = []
    semantic_error_count = 0
    obvious_error_count = 0
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
            perplexity = substitution_raw_perplexities[subst_cursor]
            normalized_perplexity = substitution_norm_perplexities[subst_cursor]
            subst_cursor += 1

            if embedder.is_model_backed:
                similarity: float | None = embedder.similarity(item.predicted_word, item.ground_truth_word)
                error_type = classify_substitution(similarity, cfg.semantic_threshold)
            else:
                similarity = None
                error_type = None

            score_value = composite_score(
                normalized_perplexity,
                similarity,
                perplexity_weight=cfg.perplexity_weight,
                semantic_weight=cfg.semantic_weight,
            )

            if error_type == "SEMANTIC":
                semantic_error_count += 1
            else:
                obvious_error_count += 1

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
        wer=wer,
        metadata=metadata or {},
    )


def score_batch(
    pairs: list[tuple[str, str]] | list[tuple[str, str, dict]],
    scorer_version: str = "1.0",
    metadata: dict[str, object] | None = None,
    config: ScorerConfig | None = None,
) -> list[SemanticErrorReport]:
    cfg = config or ScorerConfig(scorer_version=scorer_version)
    reports = []
    for pair in pairs:
        pred, gt = pair[0], pair[1]
        pair_meta: dict[str, object] = dict(pair[2]) if len(pair) > 2 else (metadata or {})  # type: ignore[misc]
        reports.append(
            score(
                predicted=pred,
                ground_truth=gt,
                scorer_version=cfg.scorer_version,
                metadata=pair_meta,
                config=cfg,
            )
        )
    return reports

