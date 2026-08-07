"""Execution d'une campagne texte (§14.1).

Chaque campagne ecrit ses predictions brutes, ses metriques et son environnement
dans `results/runs/<run_id>/`. Aucun chiffre publie ensuite ne peut exister sans
ce dossier : c'est la condition de tracabilite du §37.8.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ivr_bench.benchmark import environment as env
from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import SessionContext
from ivr_bench.domain.paths import results_dir
from ivr_bench.generators.corpus import load_split
from ivr_bench.generators.utterances import GeneratedCase
from ivr_bench.metrics.routing import Evaluation
from ivr_bench.routers import create

# Patient authentifie par la session : il n'est jamais extrait de la parole.
BENCHMARK_PATIENT = "patient_00002"


@dataclass(frozen=True)
class SuiteSelection:
    """Cas retenus, et ce qui a ete ecarte."""

    cases: list[GeneratedCase]
    total_available: int
    per_function: int | None

    @property
    def is_restricted(self) -> bool:
        return len(self.cases) < self.total_available


def stratified_sample(
    cases: list[GeneratedCase], per_function: int | None, seed: int
) -> SuiteSelection:
    """Echantillon equilibre par fonction.

    Une restriction de couverture est un fait a publier, pas a masquer : la
    selection conserve de quoi la declarer dans le manifeste du run (§18.7).
    """
    if per_function is None:
        return SuiteSelection(cases=list(cases), total_available=len(cases), per_function=None)

    grouped: dict[str, list[GeneratedCase]] = defaultdict(list)
    for case in cases:
        grouped[case.expected.tool_name].append(case)

    rng = random.Random(seed)
    selected: list[GeneratedCase] = []
    for function in sorted(grouped):
        pool = grouped[function]
        selected.extend(rng.sample(pool, min(per_function, len(pool))))
    return SuiteSelection(cases=selected, total_available=len(cases), per_function=per_function)


def new_run_id(architecture: str, seed: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{architecture}_seed{seed}"


def run_text_benchmark(
    architecture: str,
    router_options: dict[str, Any] | None = None,
    split: str = "test",
    per_function: int | None = None,
    seed: int = 42,
    config_path: Path | None = None,
    command: str = "",
) -> Path:
    """Execute une campagne texte et renvoie le dossier du run."""
    catalog = default_catalog()
    tools = list(catalog.functions)
    session = SessionContext(patient_id=BENCHMARK_PATIENT)

    selection = stratified_sample(load_split(split), per_function, seed)
    run_id = new_run_id(architecture, seed)
    directory = results_dir() / "runs" / run_id

    context = env.capture(
        run_id=run_id,
        seeds=[seed],
        command=command or f"ivr-bench benchmark text --architectures {architecture}",
        config_path=config_path,
        dataset_paths={
            split: results_dir().parent / "data" / "generated" / split / f"{split}.jsonl"
        },
    )

    router = create(architecture, **(router_options or {}))
    evaluation = Evaluation()
    directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for case in selection.cases:
            prediction = router.predict(case.utterance, session, tools)
            evaluation.observe(
                expected_tool=case.expected.tool_name,
                expected_arguments=case.expected.arguments,
                predicted_tool=prediction.tool_name,
                predicted_arguments=prediction.arguments,
                suite=case.suite,
                latency_ms=prediction.latency_ms,
                validity=str(prediction.metadata.get("validity", "native")),
            )
            handle.write(
                json.dumps(
                    {
                        "id": case.id,
                        "suite": case.suite,
                        "utterance": case.utterance,
                        "expected": case.expected.model_dump(),
                        "predicted": {
                            "tool_name": prediction.tool_name,
                            "arguments": prediction.arguments,
                            "confidence": prediction.confidence,
                        },
                        "latency_ms": round(prediction.latency_ms, 3),
                        "metadata": prediction.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    duration = time.perf_counter() - started

    metrics = evaluation.to_dict()
    metrics["architecture"] = architecture
    metrics["split"] = split
    # La couverture reelle est declaree, y compris quand elle est partielle.
    metrics["coverage"] = {
        "evaluated": len(selection.cases),
        "available": selection.total_available,
        "per_function_cap": selection.per_function,
        "restricted": selection.is_restricted,
    }
    (directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    env.write(context, directory, duration_s=duration)
    return directory
