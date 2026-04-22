from misnomer.classifier import classify_substitution
from misnomer.composite import composite_score, normalize_perplexities
from misnomer.models.lm import LMScorer
from misnomer.config import ScorerConfig


def test_classify_semantic() -> None:
    assert classify_substitution(0.9) == "SEMANTIC"
    assert classify_substitution(0.5) == "SEMANTIC"


def test_classify_obvious() -> None:
    assert classify_substitution(0.4) == "OBVIOUS"
    assert classify_substitution(0.0) == "OBVIOUS"


def test_classify_custom_threshold() -> None:
    assert classify_substitution(0.6, threshold=0.7) == "OBVIOUS"
    assert classify_substitution(0.8, threshold=0.7) == "SEMANTIC"


def test_normalize_perplexities_empty() -> None:
    assert normalize_perplexities([]) == []


def test_normalize_perplexities_uniform() -> None:
    result = normalize_perplexities([3.0, 3.0, 3.0])
    assert result == [0.0, 0.0, 0.0]


def test_normalize_perplexities_range() -> None:
    result = normalize_perplexities([1.0, 3.0, 5.0])
    assert result[0] == 0.0
    assert result[-1] == 1.0
    assert 0.0 < result[1] < 1.0


def test_composite_score_clipped() -> None:
    assert composite_score(1.0, 1.0) <= 1.0
    assert composite_score(0.0, 0.0) >= 0.0


def test_lm_scorer_proxy_returns_all_indices() -> None:
    cfg = ScorerConfig(local_files_only=True, allow_download=False)
    lm = LMScorer(cfg)
    words = ["The", "house", "stood", "in", "the", "field"]
    perplexities = lm.word_perplexities(words)
    assert set(perplexities.keys()) == set(range(len(words)))
    assert all(isinstance(v, float) and v > 0 for v in perplexities.values())


def test_lm_scorer_rare_word_higher_perplexity() -> None:
    cfg = ScorerConfig(local_files_only=True, allow_download=False)
    lm = LMScorer(cfg)
    # When using the proxy, rare words get higher perplexity.
    words = ["the", "the", "the", "serendipitous"]
    perplexities = lm.word_perplexities(words)
    # "serendipitous" appears once vs "the" appearing three times.
    assert perplexities[3] > perplexities[0]
