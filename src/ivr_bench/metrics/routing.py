"""Metriques de routage et d'arguments (§17.1, §17.2).

Deux principes gouvernent ce module :

1. **Un argument halluciné coûte plus cher qu'un argument absent** (§12). Ne pas
   savoir est un état honnête que le dialogue peut rattraper en posant une
   question ; inventer une date fait prendre un rendez-vous le mauvais jour.
2. **Aucun agrégat sans détail** (§18.9). Chaque chiffre global s'accompagne de
   sa décomposition par fonction et par sous-suite.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArgumentTally:
    """Comptes bruts servant aux metriques d'arguments."""

    exact: int = 0
    total: int = 0
    hallucinated: int = 0
    missing: int = 0
    correctly_absent: int = 0

    @property
    def exact_match(self) -> float | None:
        return self.exact / self.total if self.total else None

    @property
    def hallucination_rate(self) -> float | None:
        return self.hallucinated / self.total if self.total else None

    @property
    def missing_rate(self) -> float | None:
        return self.missing / self.total if self.total else None


@dataclass
class RoutingTally:
    """Comptes bruts du routage."""

    correct: int = 0
    total: int = 0
    predicted: Counter[str] = field(default_factory=Counter)
    expected: Counter[str] = field(default_factory=Counter)
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None


def _normalise(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).lower().split())
    return text or None


def score_arguments(expected: dict[str, Any], produced: dict[str, Any]) -> ArgumentTally:
    """Compare les arguments attendus et produits, cle par cle."""
    tally = ArgumentTally()
    for key in expected:
        want = _normalise(expected.get(key))
        got = _normalise(produced.get(key))
        tally.total += 1
        if want == got:
            tally.exact += 1
            if want is None:
                tally.correctly_absent += 1
        elif want is None:
            # Le modele a rempli un champ que l'enonce n'exprimait pas.
            tally.hallucinated += 1
        elif got is None:
            tally.missing += 1
    # Une cle produite hors du schema est une hallucination franche ; la
    # validation l'a deja rejetee, on la compte ici pour la metrique.
    for key in produced:
        if key not in expected:
            tally.total += 1
            tally.hallucinated += 1
    return tally


@dataclass
class Evaluation:
    """Resultat agrege d'une campagne, decompose par fonction et sous-suite."""

    routing: RoutingTally = field(default_factory=RoutingTally)
    arguments: ArgumentTally = field(default_factory=ArgumentTally)
    by_function: dict[str, RoutingTally] = field(default_factory=lambda: defaultdict(RoutingTally))
    by_suite: dict[str, RoutingTally] = field(default_factory=lambda: defaultdict(RoutingTally))
    validity: Counter[str] = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)

    def observe(
        self,
        expected_tool: str,
        expected_arguments: dict[str, Any],
        predicted_tool: str | None,
        predicted_arguments: dict[str, Any],
        suite: str,
        latency_ms: float,
        validity: str = "native",
    ) -> None:
        correct = predicted_tool == expected_tool
        for tally in (self.routing, self.by_function[expected_tool], self.by_suite[suite]):
            tally.total += 1
            tally.correct += int(correct)
            tally.expected[expected_tool] += 1
            if predicted_tool is not None:
                tally.predicted[predicted_tool] += 1
        self.routing.confusion[(expected_tool, predicted_tool or "aucune")] += 1

        # Les arguments ne sont comptes que lorsque la fonction est correcte :
        # comparer les arguments de deux fonctions differentes n'a pas de sens.
        if correct:
            partial = score_arguments(expected_arguments, predicted_arguments)
            self.arguments.exact += partial.exact
            self.arguments.total += partial.total
            self.arguments.hallucinated += partial.hallucinated
            self.arguments.missing += partial.missing
            self.arguments.correctly_absent += partial.correctly_absent

        self.validity[validity] += 1
        self.latencies_ms.append(latency_ms)

    # -- restitution --------------------------------------------------------

    def macro_f1(self) -> float | None:
        """F1 macro sur les fonctions, insensible au desequilibre des classes."""
        if not self.routing.total:
            return None
        scores: list[float] = []
        for function, tally in self.by_function.items():
            true_positive = tally.correct
            predicted_total = sum(
                other.predicted.get(function, 0) for other in self.by_function.values()
            )
            precision = true_positive / predicted_total if predicted_total else 0.0
            recall = true_positive / tally.total if tally.total else 0.0
            scores.append(
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
        return sum(scores) / len(scores) if scores else None

    def recall_for(self, function: str) -> float | None:
        tally = self.by_function.get(function)
        return tally.accuracy if tally else None

    def percentile(self, fraction: float) -> float | None:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        position = min(int(len(ordered) * fraction), len(ordered) - 1)
        return ordered[position]

    def to_dict(self) -> dict[str, Any]:
        """Restitution complete : aucun agregat sans son detail."""
        return {
            "tool_accuracy": self.routing.accuracy,
            "macro_f1": self.macro_f1(),
            # Metrique de securite prioritaire du §17.1.
            "emergency_handoff_recall": self.recall_for("emergency_handoff"),
            "no_tool_recall": self.recall_for("no_tool"),
            "argument_exact_match": self.arguments.exact_match,
            "hallucinated_argument_rate": self.arguments.hallucination_rate,
            "missing_argument_rate": self.arguments.missing_rate,
            "output_validity": dict(self.validity),
            "latency_ms": {
                "p50": self.percentile(0.50),
                "p95": self.percentile(0.95),
                "p99": self.percentile(0.99),
                "mean": (
                    sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None
                ),
            },
            "count": self.routing.total,
            "by_function": {
                name: {"accuracy": tally.accuracy, "count": tally.total}
                for name, tally in sorted(self.by_function.items())
            },
            "by_suite": {
                name: {"accuracy": tally.accuracy, "count": tally.total}
                for name, tally in sorted(self.by_suite.items())
            },
            "confusion": {
                f"{expected} -> {predicted}": count
                for (expected, predicted), count in sorted(self.routing.confusion.items())
            },
        }
