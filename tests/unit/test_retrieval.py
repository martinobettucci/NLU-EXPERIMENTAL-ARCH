"""Tests des prototypes, de l'agregation et de l'index (§10.6, §11).

Ces tests n'ont besoin d'aucun poids : ils travaillent sur des vecteurs
construits a la main, ce qui permet de verifier la mecanique de scoring
independamment de la qualite d'un encodeur.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ivr_bench.embeddings.base import l2_normalize
from ivr_bench.embeddings.projection import LearnedProjection
from ivr_bench.retrieval import prototypes
from ivr_bench.retrieval.aggregation import Neighbour, aggregate
from ivr_bench.retrieval.index import IndexMetadata, SemanticIndex


def _cluster(centre: list[float], count: int, spread: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.asarray(centre, dtype=np.float32)
    noise = rng.normal(0.0, spread, size=(count, base.shape[0])).astype(np.float32)
    return l2_normalize(base + noise)


# -- prototypes -------------------------------------------------------------


def test_normalisation_is_idempotent() -> None:
    vectors = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    once = l2_normalize(vectors)
    assert np.allclose(np.linalg.norm(once, axis=1), 1.0)
    assert np.allclose(l2_normalize(once), once)


def test_zero_vector_does_not_produce_nan() -> None:
    result = l2_normalize(np.zeros((1, 4), dtype=np.float32))
    assert not np.isnan(result).any()


def test_centroid_collapses_distinct_modes() -> None:
    """L'hypothese du §10.6 : un centroide unique detruit les modes minoritaires."""
    first = _cluster([1.0, 0.0, 0.0], 40, 0.05, seed=1)
    second = _cluster([0.0, 1.0, 0.0], 40, 0.05, seed=2)
    both = np.vstack([first, second])

    single = prototypes.build(both, "centroid")
    multiple = prototypes.build(both, "kmeans", count=2, seed=3)

    # Le centroide unique s'eloigne des deux modes ; deux prototypes en
    # conservent au moins un tres proche de chacun.
    assert float(np.max(first @ single[0])) < 0.85
    assert float(np.max(first @ multiple.T)) > 0.95
    assert float(np.max(second @ multiple.T)) > 0.95


def test_kmeans_is_deterministic() -> None:
    vectors = _cluster([1.0, 0.0, 0.0], 60, 0.2, seed=4)
    assert np.allclose(
        prototypes.build(vectors, "kmeans", count=4, seed=7),
        prototypes.build(vectors, "kmeans", count=4, seed=7),
    )


def test_medoids_are_real_utterances() -> None:
    """Un medoid existe dans le corpus ; un centroide peut tomber dans le vide."""
    vectors = np.vstack(
        [_cluster([1.0, 0.0, 0.0], 30, 0.1, 5), _cluster([0.0, 1.0, 0.0], 30, 0.1, 6)]
    )
    chosen = prototypes.build(vectors, "medoids", count=2, seed=8)
    for prototype in chosen:
        assert np.isclose(vectors @ prototype, 1.0, atol=1e-5).any()


def test_all_strategy_keeps_every_vector() -> None:
    vectors = _cluster([1.0, 0.0], 12, 0.1, seed=9)
    assert prototypes.build(vectors, "all").shape == vectors.shape


def test_requesting_more_prototypes_than_vectors_is_safe() -> None:
    vectors = _cluster([1.0, 0.0], 3, 0.1, seed=10)
    assert prototypes.build(vectors, "kmeans", count=10).shape[0] <= 3


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="strategie"):
        prototypes.build(_cluster([1.0, 0.0], 5, 0.1, 11), "mediane")  # type: ignore[arg-type]


# -- agregation -------------------------------------------------------------


def test_score_combines_best_mean_and_support() -> None:
    neighbours = [
        Neighbour("a", 0.9),
        Neighbour("a", 0.8),
        Neighbour("a", 0.7),
        Neighbour("b", 0.6),
    ]
    ranking = aggregate(neighbours)
    expected = 0.55 * 0.9 + 0.35 * (0.9 + 0.8 + 0.7) / 3 + 0.10 * (3 / 4)
    assert ranking.scores["a"] == pytest.approx(expected)


def test_support_alone_does_not_beat_a_much_closer_match() -> None:
    """Le score n'est pas un vote : un voisin nettement plus proche l'emporte."""
    neighbours = [Neighbour("close", 0.95)] + [Neighbour("many", 0.40) for _ in range(10)]
    assert aggregate(neighbours).top(1) == ["close"]


def test_margin_reports_the_gap_not_the_rank() -> None:
    tight = aggregate([Neighbour("a", 0.80), Neighbour("b", 0.79)])
    clear = aggregate([Neighbour("a", 0.95), Neighbour("b", 0.30)])
    assert tight.uncertainty.margin < clear.uncertainty.margin


def test_purity_and_entropy_track_neighbourhood_agreement() -> None:
    unanimous = aggregate([Neighbour("a", 0.9) for _ in range(5)])
    divided = aggregate([Neighbour(name, 0.9) for name in ("a", "b", "c", "d", "e")])
    assert unanimous.uncertainty.purity == 1.0
    assert unanimous.uncertainty.entropy == 0.0
    assert divided.uncertainty.purity < unanimous.uncertainty.purity
    assert divided.uncertainty.entropy > unanimous.uncertainty.entropy


def test_empty_neighbourhood_yields_no_score() -> None:
    ranking = aggregate([])
    assert ranking.scores == {}
    assert ranking.top(2) == []


# -- projection apprise -----------------------------------------------------


def test_projection_reduces_dimension_and_renormalises() -> None:
    vectors = l2_normalize(np.random.default_rng(12).normal(size=(200, 32)).astype(np.float32))
    projection = LearnedProjection(8).fit(vectors)
    reduced = projection.transform(vectors)
    assert reduced.shape == (200, 8)
    assert np.allclose(np.linalg.norm(reduced, axis=1), 1.0, atol=1e-5)


def test_projection_refuses_to_fit_on_too_few_examples() -> None:
    """Ajuster 64 dimensions sur 10 exemples produirait une reduction fantaisiste."""
    with pytest.raises(ValueError, match="trop peu"):
        LearnedProjection(64).fit(np.zeros((10, 128), dtype=np.float32))


def test_projection_must_be_fitted_before_use() -> None:
    with pytest.raises(RuntimeError, match="non ajustee"):
        LearnedProjection(4).transform(np.zeros((2, 16), dtype=np.float32))


def test_projection_survives_a_round_trip(tmp_path: Path) -> None:
    vectors = l2_normalize(np.random.default_rng(13).normal(size=(100, 16)).astype(np.float32))
    projection = LearnedProjection(4).fit(vectors)
    projection.save(tmp_path / "projection.npz")
    reloaded = LearnedProjection.load(tmp_path / "projection.npz")
    assert np.allclose(projection.transform(vectors), reloaded.transform(vectors))


# -- index ------------------------------------------------------------------


def _index() -> SemanticIndex:
    vectors = l2_normalize(
        np.asarray([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    )
    metadata = IndexMetadata(
        encoder="test",
        dimension=3,
        strategy="all",
        prototypes_per_function=1,
        seed=42,
        source_cases=3,
        built_at="2026-03-02T09:00:00+00:00",
        content_sha256="0" * 64,
    )
    return SemanticIndex(vectors, ["alpha", "alpha", "beta"], metadata)


def test_search_returns_neighbours_in_descending_similarity() -> None:
    neighbours = _index().search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)
    similarities = [item.similarity for item in neighbours]
    assert similarities == sorted(similarities, reverse=True)
    assert neighbours[0].function == "alpha"


def test_ranking_prefers_the_nearest_function() -> None:
    ranking = _index().rank(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), top_k=3)
    assert ranking.top(1) == ["beta"]


def test_owner_count_must_match_prototype_count() -> None:
    with pytest.raises(ValueError, match="proprietaire"):
        SemanticIndex(
            np.zeros((2, 3), dtype=np.float32),
            ["alpha"],
            _index().metadata,
        )


def test_index_survives_a_round_trip(tmp_path: Path) -> None:
    original = _index()
    original.save(tmp_path)
    reloaded = SemanticIndex.load(tmp_path)

    query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    assert reloaded.size == original.size
    assert reloaded.functions == original.functions
    assert reloaded.metadata == original.metadata
    assert reloaded.rank(query).scores == original.rank(query).scores
