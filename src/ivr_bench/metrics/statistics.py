"""Protocole statistique (§18).

Deux architectures separees de quelques points sur 84 cas ne sont pas
distinguables. Publier ces points sans intervalle laisserait croire le
contraire. Ce module fournit l'intervalle et le test apparie qui permettent de
dire « la difference n'est pas etablie » — une conclusion aussi utile que son
inverse.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """Estimation ponctuelle et intervalle de confiance."""

    estimate: float
    low: float
    high: float
    level: float = 0.95

    def __str__(self) -> str:
        return f"{self.estimate:.1%} [{self.low:.1%} – {self.high:.1%}]"


def bootstrap_proportion(
    successes: list[bool],
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 42,
) -> Interval:
    """Intervalle bootstrap percentile sur une proportion.

    Le tirage est seede : deux lectures du meme run donnent le meme intervalle,
    sinon la borne publiee dependrait du moment de la lecture.
    """
    if not successes:
        raise ValueError("aucune observation")

    count = len(successes)
    estimate = sum(successes) / count
    rng = random.Random(seed)

    draws: list[float] = []
    for _ in range(resamples):
        drawn = sum(successes[rng.randrange(count)] for _ in range(count))
        draws.append(drawn / count)
    draws.sort()

    tail = (1.0 - level) / 2.0
    low = draws[int(tail * resamples)]
    high = draws[min(int((1.0 - tail) * resamples), resamples - 1)]
    return Interval(estimate=estimate, low=low, high=high, level=level)


@dataclass(frozen=True)
class McNemarResult:
    """Comparaison appariee de deux architectures sur les memes cas."""

    only_first_correct: int
    only_second_correct: int
    statistic: float
    p_value: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    def verdict(self, first: str, second: str) -> str:
        if not self.is_significant:
            # Formulation deliberee : l'absence de preuve n'est pas une preuve
            # d'equivalence.
            return f"difference non etablie entre {first} et {second} (p = {self.p_value:.3f})"
        better = first if self.only_first_correct > self.only_second_correct else second
        return f"{better} l'emporte (p = {self.p_value:.3f})"


def _binomial_two_sided(successes: int, trials: int) -> float:
    """Test binomial exact a p = 0,5, utilise quand les effectifs sont faibles."""
    if trials == 0:
        return 1.0
    coefficients = [math.comb(trials, k) for k in range(trials + 1)]
    total = float(sum(coefficients))
    observed = coefficients[successes]
    # Somme des issues au moins aussi extremes que celle observee.
    tail = sum(value for value in coefficients if value <= observed + 1e-12)
    return min(1.0, tail / total)


def mcnemar(first_correct: list[bool], second_correct: list[bool]) -> McNemarResult:
    """Test de McNemar sur deux series appariees de reussites."""
    if len(first_correct) != len(second_correct):
        raise ValueError("les deux series doivent porter sur les memes cas")

    only_first = sum(1 for a, b in zip(first_correct, second_correct, strict=True) if a and not b)
    only_second = sum(1 for a, b in zip(first_correct, second_correct, strict=True) if b and not a)
    discordant = only_first + only_second

    if discordant == 0:
        return McNemarResult(only_first, only_second, statistic=0.0, p_value=1.0)

    # Sous 25 desaccords, l'approximation du chi2 n'est pas fiable : on bascule
    # sur le test exact plutot que de publier une valeur trop confiante.
    if discordant < 25:
        return McNemarResult(
            only_first,
            only_second,
            statistic=float(min(only_first, only_second)),
            p_value=_binomial_two_sided(min(only_first, only_second), discordant),
        )

    # Correction de continuite d'Edwards.
    statistic = (abs(only_first - only_second) - 1.0) ** 2 / discordant
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    return McNemarResult(only_first, only_second, statistic=statistic, p_value=p_value)
