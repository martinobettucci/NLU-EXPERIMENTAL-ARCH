"""Tests du protocole statistique (§18)."""

from __future__ import annotations

import pytest

from ivr_bench.metrics.statistics import bootstrap_proportion, mcnemar


def test_interval_brackets_the_estimate() -> None:
    successes = [True] * 70 + [False] * 30
    interval = bootstrap_proportion(successes, seed=1)
    assert interval.low <= interval.estimate <= interval.high
    assert interval.estimate == pytest.approx(0.70)


def test_interval_is_reproducible() -> None:
    """Une borne publiee ne doit pas dependre du moment de la lecture."""
    successes = [True] * 40 + [False] * 44
    first = bootstrap_proportion(successes, seed=7)
    second = bootstrap_proportion(successes, seed=7)
    assert (first.low, first.high) == (second.low, second.high)


def test_small_samples_give_wider_intervals() -> None:
    """C'est tout l'interet : 84 cas ne permettent pas les memes conclusions."""
    small = bootstrap_proportion([True] * 12 + [False] * 12, seed=3)
    large = bootstrap_proportion([True] * 1000 + [False] * 1000, seed=3)
    assert (small.high - small.low) > (large.high - large.low)


def test_empty_sample_is_rejected() -> None:
    with pytest.raises(ValueError, match="aucune observation"):
        bootstrap_proportion([])


def test_identical_architectures_show_no_difference() -> None:
    series = [True, False, True, True, False] * 10
    result = mcnemar(series, series)
    assert result.p_value == 1.0
    assert not result.is_significant


def test_a_small_gap_is_not_declared_significant() -> None:
    """Deux points d'ecart sur un petit echantillon ne prouvent rien."""
    first = [True] * 42 + [False] * 42
    second = [True] * 40 + [False] * 44
    result = mcnemar(first, second)
    assert not result.is_significant
    assert "non etablie" in result.verdict("A", "B")


def test_a_systematic_advantage_is_detected() -> None:
    first = [True] * 60 + [False] * 24
    second = [False] * 84
    result = mcnemar(first, second)
    assert result.is_significant
    assert result.verdict("A", "B").startswith("A l'emporte")


def test_mismatched_series_are_rejected() -> None:
    with pytest.raises(ValueError, match="memes cas"):
        mcnemar([True, False], [True])


def test_exact_test_is_used_when_disagreements_are_few() -> None:
    """Sous 25 desaccords, le chi2 donnerait une valeur trop confiante."""
    first = [True] * 3 + [False] * 81
    second = [False] * 84
    result = mcnemar(first, second)
    assert result.only_first_correct == 3
    assert result.only_second_correct == 0
    assert result.p_value > 0.05
