"""Graphiques statiques (§31).

Chaque figure est ecrite en PNG et en SVG, et ses donnees sources vivent dans
`results/tables/`. Une figure dont on ne peut pas relire les chiffres n'est pas
un resultat.

Les barres d'erreur ne sont pas decoratives : sur 84 cas, l'intervalle est plus
large que la plupart des ecarts entre architectures, et une figure sans lui
laisserait lire un classement qui n'existe pas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Backend sans affichage : la campagne tourne sans serveur graphique.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ivr_bench.domain.paths import results_dir
from ivr_bench.reporting.tables import suite_frame, summary_frame


def charts_dir() -> Path:
    return results_dir() / "charts"


def _save(figure: Figure, name: str) -> list[Path]:
    directory = charts_dir()
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in ("png", "svg"):
        path = directory / f"{name}.{suffix}"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


def accuracy_with_intervals(frame: pd.DataFrame) -> list[Path]:
    """Exactitude par architecture, avec son intervalle de confiance."""
    data = frame.dropna(subset=["tool_accuracy"]).sort_values("tool_accuracy")
    figure, axes = plt.subplots(figsize=(9, 5))

    low = data["tool_accuracy"] - data["accuracy_ci_low"]
    high = data["accuracy_ci_high"] - data["tool_accuracy"]
    axes.barh(data["architecture"], data["tool_accuracy"], xerr=[low, high], capsize=4)
    axes.set_xlabel("Exactitude de la fonction (intervalle bootstrap 95 %)")
    axes.set_xlim(0, 1)
    # La couverture figure sur chaque barre : comparer 84 cas a 2 088 sans le
    # dire serait trompeur.
    for position, (_, row) in enumerate(data.iterrows()):
        axes.text(0.01, position, f"{row['cases']} cas", va="center", fontsize=8, color="white")
    axes.set_title("Exactitude par architecture")
    return _save(figure, "accuracy_by_architecture")


def accuracy_versus_latency(frame: pd.DataFrame) -> list[Path]:
    """Precision contre latence p95 : la vue de compromis du §31.1."""
    data = frame.dropna(subset=["tool_accuracy", "latency_p95_ms"])
    figure, axes = plt.subplots(figsize=(8, 5.5))
    axes.scatter(data["latency_p95_ms"], data["tool_accuracy"], s=60)
    for _, row in data.iterrows():
        axes.annotate(
            row["architecture"],
            (row["latency_p95_ms"], row["tool_accuracy"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    axes.set_xscale("symlog")
    axes.set_xlabel("Latence p95 (ms, echelle logarithmique)")
    axes.set_ylabel("Exactitude de la fonction")
    axes.set_ylim(0, 1)
    axes.set_title("Precision contre latence (CPU)")
    return _save(figure, "accuracy_vs_latency")


def safety_recall(frame: pd.DataFrame) -> list[Path]:
    """Rappel de `emergency_handoff`, metrique de securite prioritaire."""
    data = frame.dropna(subset=["emergency_handoff_recall"]).sort_values("emergency_handoff_recall")
    figure, axes = plt.subplots(figsize=(9, 4.5))
    axes.barh(data["architecture"], data["emergency_handoff_recall"], color="#b03a2e")
    axes.set_xlim(0, 1)
    axes.set_xlabel("Rappel de emergency_handoff")
    axes.set_title("Metrique de securite prioritaire")
    return _save(figure, "safety_recall")


def accuracy_by_suite(frame: pd.DataFrame) -> list[Path]:
    """Decomposition par sous-suite : ou chaque architecture cede."""
    if frame.empty:
        return []
    pivot = frame.pivot_table(
        index="suite", columns="architecture", values="accuracy", aggfunc="mean"
    )
    figure, axes = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=axes, width=0.82)
    axes.set_ylabel("Exactitude")
    axes.set_ylim(0, 1)
    axes.set_xlabel("Sous-suite")
    axes.set_title("Exactitude par sous-suite")
    axes.legend(fontsize=7, ncol=2)
    return _save(figure, "accuracy_by_suite")


def build_charts() -> list[Path]:
    """Produit toutes les figures disponibles."""
    summary = summary_frame()
    if summary.empty:
        return []

    written: list[Path] = []
    written.extend(accuracy_with_intervals(summary))
    written.extend(accuracy_versus_latency(summary))
    written.extend(safety_recall(summary))
    written.extend(accuracy_by_suite(suite_frame()))
    return written
