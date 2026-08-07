"""Execution d'une campagne texte (§14.1).

Chaque campagne ecrit ses predictions brutes, ses metriques et son environnement
dans `results/runs/<run_id>/`. Aucun chiffre publie ensuite ne peut exister sans
ce dossier : c'est la condition de tracabilite du §37.8.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterator
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


@contextlib.contextmanager
def exclusive_campaign() -> Iterator[None]:
    """Interdit deux campagnes simultanees sur la meme machine.

    Ce n'est pas une question de vitesse. Les latences p50 et p95 font partie
    des resultats publies ; mesurees pendant qu'un entrainement occupe les
    memes coeurs, elles decrivent la contention, pas l'architecture. Le verrou
    rend la mise en file obligatoire plutot que dependante de la vigilance de
    l'appelant.
    """
    lock_path = results_dir() / "runs" / ".campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(
                "une campagne est deja en cours sur cette machine. Les latences "
                "mesurees en concurrence ne seraient pas exploitables : attendez "
                "la fin de la campagne precedente."
            ) from None
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


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

    # Charge au demarrage : une campagne lancee sur une machine deja occupee
    # produit des latences ininterpretables, et le fait doit rester lisible
    # dans le run plutot que d'etre devine apres coup.
    load_before = os.getloadavg()[0]
    # Certaines architectures analysent en lot, modele charge une seule fois.
    # La preparation est faite AVANT le chronometre : elle appartient au cout de
    # demarrage, pas a la latence d'inference.
    prepare = getattr(router, "prepare", None)
    if callable(prepare):
        prepare([case.utterance for case in selection.cases])

    # Passe de chauffe, exclue de la mesure.
    #
    # Les poids se chargent a la premiere inference, pas a la construction du
    # routeur : sans cette passe, le premier enonce porte le chargement du
    # modele et la compilation. Mesure avant correction : 22 036 ms pour le
    # premier appel de Needle contre 4 772 de mediane, 11 562 ms pour le premier
    # appel du classifieur contre 95. C'est un cout de demarrage, il n'a rien a
    # faire dans une distribution de latence d'inference.
    #
    # La chauffe utilise le premier cas reel plutot qu'une phrase inventee :
    # certaines architectures analysent leur lot a l'avance et n'accepteraient
    # pas un enonce absent de ce lot.
    warmup_ms: float | None = None
    if selection.cases:
        warmup_started = time.perf_counter()
        router.predict(selection.cases[0].utterance, session, tools)
        warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

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
    # Le cout de demarrage est publie, pas efface : c'est lui qui decide si une
    # architecture est deployable sur une machine qui redemarre souvent.
    metrics["warmup_ms"] = round(warmup_ms, 3) if warmup_ms is not None else None
    metrics["machine_load"] = {
        "before": round(load_before, 2),
        "after": round(os.getloadavg()[0], 2),
        "cpu_count": os.cpu_count(),
    }
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

    # Marqueur ecrit en dernier. Une campagne interrompue laisse ses predictions
    # partielles sur le disque ; sans ce temoin, un rapport ulterieur pourrait
    # les prendre pour un resultat. Le §37.9 interdit de publier un run
    # incomplet : on le rend donc reconnaissable plutot que plausible.
    (directory / "COMPLETE").write_text(
        f"{context.run_id}\n{context.timestamp}\n", encoding="utf-8"
    )
    return directory


def is_complete(directory: Path) -> bool:
    """Un run n'est exploitable que s'il a ete mene a son terme.

    Le critere porte sur les artefacts, pas sur le temoin : `metrics.json` et
    `environment.json` ne sont ecrits qu'a la fin, donc leur presence suffit a
    prouver l'aboutissement. Exiger le marqueur ferait passer pour incomplets
    les runs anterieurs a son introduction — et le §29.3 interdit d'ecraser les
    runs historiques.
    """
    return all(
        (directory / name).is_file()
        for name in ("metrics.json", "environment.json", "predictions.jsonl")
    )


def iter_runs(complete_only: bool = True) -> list[Path]:
    """Runs disponibles, les incomplets etant ecartes explicitement."""
    root = results_dir() / "runs"
    if not root.is_dir():
        return []
    found = sorted(path for path in root.iterdir() if path.is_dir())
    return [path for path in found if not complete_only or is_complete(path)]


def incomplete_runs() -> list[Path]:
    """Runs interrompus, a signaler plutot qu'a ignorer en silence."""
    return [path for path in iter_runs(complete_only=False) if not is_complete(path)]
