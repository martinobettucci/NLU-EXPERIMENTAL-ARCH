"""Agregation des voisins en scores par fonction (§11).

Le classement ne se fait pas sur la meilleure similarite seule : un unique
voisin tres proche peut etre un accident de formulation. On combine la
meilleure similarite, la solidite du voisinage immediat et le soutien global,
puis on expose separement les signaux d'incertitude — dont le delta cosinus, qui
est un indicateur de doute, jamais le score de classement.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Ponderations par defaut du §11, calibrees sur la validation uniquement.
WEIGHT_BEST = 0.55
WEIGHT_TOP3 = 0.35
WEIGHT_SUPPORT = 0.10


@dataclass(frozen=True)
class Neighbour:
    """Un prototype retrouve, avec la fonction qui le possede."""

    function: str
    similarity: float


@dataclass(frozen=True)
class Uncertainty:
    """Signaux d'incertitude accompagnant un classement."""

    top_score: float
    margin: float
    purity: float
    entropy: float


@dataclass(frozen=True)
class Ranking:
    """Fonctions classees et signaux associes."""

    scores: dict[str, float]
    uncertainty: Uncertainty
    neighbours: tuple[Neighbour, ...] = field(default=())

    @property
    def ordered(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda item: (-item[1], item[0]))

    def top(self, count: int) -> list[str]:
        return [name for name, _ in self.ordered[:count]]


def _entropy(weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    probabilities = [weight / total for weight in weights if weight > 0.0]
    if len(probabilities) <= 1:
        return 0.0
    raw = -sum(p * math.log(p) for p in probabilities)
    # Normalisee entre 0 et 1 pour rester comparable d'un catalogue a l'autre.
    return raw / math.log(len(probabilities))


def aggregate(
    neighbours: list[Neighbour],
    weight_best: float = WEIGHT_BEST,
    weight_top3: float = WEIGHT_TOP3,
    weight_support: float = WEIGHT_SUPPORT,
) -> Ranking:
    """Regroupe les voisins par fonction et calcule les scores."""
    if not neighbours:
        return Ranking(scores={}, uncertainty=Uncertainty(0.0, 0.0, 0.0, 0.0))

    grouped: dict[str, list[float]] = defaultdict(list)
    for neighbour in neighbours:
        grouped[neighbour.function].append(neighbour.similarity)

    total_neighbours = len(neighbours)
    scores: dict[str, float] = {}
    for function, similarities in grouped.items():
        ranked = sorted(similarities, reverse=True)
        best = ranked[0]
        top3 = float(np.mean(ranked[:3]))
        # Le soutien est deja borne entre 0 et 1 : c'est une part du voisinage.
        support = len(ranked) / total_neighbours
        scores[function] = weight_best * best + weight_top3 * top3 + weight_support * support

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_function, best_score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0

    purity = len(grouped[best_function]) / total_neighbours
    uncertainty = Uncertainty(
        top_score=best_score,
        margin=best_score - runner_up,
        purity=purity,
        entropy=_entropy([len(values) for values in grouped.values()]),
    )
    return Ranking(scores=scores, uncertainty=uncertainty, neighbours=tuple(neighbours))
