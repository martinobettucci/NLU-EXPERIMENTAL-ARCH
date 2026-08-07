"""Tableaux de resultats en CSV et Parquet (§31, §37.6).

Les donnees sources des graphiques sont enregistrees a part : une figure sans
ses chiffres n'est pas verifiable. Les intervalles de confiance sont recalcules
ici depuis les predictions brutes, pas recopies d'un resume — ils doivent
pouvoir etre refaits par quiconque relit le run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ivr_bench.benchmark.runner import iter_runs
from ivr_bench.domain.paths import results_dir
from ivr_bench.metrics.calls import is_exact_call
from ivr_bench.metrics.statistics import bootstrap_proportion, mcnemar


def _load(directory: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    environment = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return metrics, environment, predictions


def summary_frame() -> pd.DataFrame:
    """Une ligne par architecture, avec son intervalle de confiance."""
    rows: list[dict[str, Any]] = []
    for directory in iter_runs(complete_only=True):
        metrics, environment, predictions = _load(directory)
        successes = [
            row["predicted"]["tool_name"] == row["expected"]["tool_name"] for row in predictions
        ]
        interval = bootstrap_proportion(successes, seed=42) if successes else None
        latency = metrics.get("latency_ms", {})
        rows.append(
            {
                "architecture": metrics["architecture"],
                "run_id": environment["run_id"],
                "git_commit": environment["git_commit"][:12],
                "git_dirty": environment["git_dirty"],
                "cases": metrics["coverage"]["evaluated"],
                "cases_available": metrics["coverage"]["available"],
                "coverage_restricted": metrics["coverage"]["restricted"],
                "tool_accuracy": metrics["tool_accuracy"],
                "accuracy_ci_low": interval.low if interval else None,
                "accuracy_ci_high": interval.high if interval else None,
                # La seule metrique qui se compte sur tout le corpus et decrit
                # ce qu'un serveur vocal peut executer tel quel.
                "exact_call_rate": (
                    sum(is_exact_call(row["expected"], row["predicted"]) for row in predictions)
                    / len(predictions)
                    if predictions
                    else None
                ),
                "macro_f1": metrics["macro_f1"],
                "emergency_handoff_recall": metrics["emergency_handoff_recall"],
                "no_tool_recall": metrics["no_tool_recall"],
                "argument_exact_match": metrics["argument_exact_match"],
                "hallucinated_argument_rate": metrics["hallucinated_argument_rate"],
                "latency_p50_ms": latency.get("p50"),
                "latency_p95_ms": latency.get("p95"),
                "native_outputs": metrics["output_validity"].get("native", 0),
                "invalid_outputs": metrics["output_validity"].get("invalid", 0),
            }
        )
    return pd.DataFrame(rows).sort_values("architecture").reset_index(drop=True)


def suite_frame() -> pd.DataFrame:
    """Decomposition par sous-suite : aucun agregat sans son detail (§18.9)."""
    rows: list[dict[str, Any]] = []
    for directory in iter_runs(complete_only=True):
        metrics, _, _ = _load(directory)
        for suite, values in metrics.get("by_suite", {}).items():
            rows.append(
                {
                    "architecture": metrics["architecture"],
                    "suite": suite,
                    "accuracy": values["accuracy"],
                    "count": values["count"],
                }
            )
    return pd.DataFrame(rows)


def function_frame() -> pd.DataFrame:
    """Decomposition par fonction metier."""
    rows: list[dict[str, Any]] = []
    for directory in iter_runs(complete_only=True):
        metrics, _, _ = _load(directory)
        for function, values in metrics.get("by_function", {}).items():
            rows.append(
                {
                    "architecture": metrics["architecture"],
                    "function": function,
                    "accuracy": values["accuracy"],
                    "count": values["count"],
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons() -> pd.DataFrame:
    """Comparaisons appariees entre architectures evaluees sur les memes cas.

    Deux architectures ne sont comparables que si elles ont vu exactement les
    memes enonces : comparer 84 cas a 2 088 produirait un classement sans
    signification.
    """
    outcomes: dict[str, dict[str, bool]] = {}
    for directory in iter_runs(complete_only=True):
        metrics, _, predictions = _load(directory)
        outcomes[metrics["architecture"]] = {
            row["id"]: row["predicted"]["tool_name"] == row["expected"]["tool_name"]
            for row in predictions
        }

    rows: list[dict[str, Any]] = []
    names = sorted(outcomes)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared = sorted(set(outcomes[first]) & set(outcomes[second]))
            if len(shared) < 20:
                continue
            left = [outcomes[first][case] for case in shared]
            right = [outcomes[second][case] for case in shared]
            result = mcnemar(left, right)
            rows.append(
                {
                    "first": first,
                    "second": second,
                    "shared_cases": len(shared),
                    "first_accuracy": sum(left) / len(left),
                    "second_accuracy": sum(right) / len(right),
                    "only_first_correct": result.only_first_correct,
                    "only_second_correct": result.only_second_correct,
                    "p_value": result.p_value,
                    "significant": result.is_significant,
                    "verdict": result.verdict(first, second),
                }
            )
    return pd.DataFrame(rows)


def tables_dir() -> Path:
    return results_dir() / "tables"


def write_tables() -> list[Path]:
    """Ecrit chaque tableau en CSV et en Parquet."""
    directory = tables_dir()
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name, frame in (
        ("summary", summary_frame()),
        ("by_suite", suite_frame()),
        ("by_function", function_frame()),
        ("paired_comparisons", paired_comparisons()),
    ):
        if frame.empty:
            continue
        csv_path = directory / f"{name}.csv"
        frame.to_csv(csv_path, index=False)
        written.append(csv_path)

        parquet_path = directory / f"{name}.parquet"
        frame.to_parquet(parquet_path, index=False)
        written.append(parquet_path)
    return written
