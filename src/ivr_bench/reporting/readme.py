"""Mise a jour de la zone generee du README (§30).

Trois regles, non negociables :

1. Seule la zone entre marqueurs est reecrite. Le reste du README appartient a
   son auteur.
2. Une metrique non mesuree affiche `non exécuté`, jamais `0`.
3. Un run marque `dirty` — depot modifie non commite — ne remplace jamais les
   chiffres publies (§28).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ivr_bench.benchmark.runner import iter_runs
from ivr_bench.domain.paths import repo_root
from ivr_bench.metrics.calls import exact_call_rate

START = "<!-- BENCHMARK_RESULTS_START -->"
END = "<!-- BENCHMARK_RESULTS_END -->"
NOT_RUN = "non exécuté"

# Toutes les architectures de la specification, y compris celles qui n'ont pas
# encore tourne : leur absence doit se voir.
ARCHITECTURES: tuple[tuple[str, str], ...] = (
    ("A0", "rules"),
    ("A1", "diet"),
    ("A2", "needle_full"),
    ("A3", "functiongemma_zero_shot"),
    ("A4", "functiongemma_tuned"),
    ("A5", "embedding_only"),
    ("A6", "hybrid_needle_top2"),
    ("A7", "hybrid_functiongemma_top2"),
    ("A8", "hybrid_adaptive"),
    # Strategies ajoutees apres recentrage sur la question reelle : d'une phrase
    # vers une fonction. Ce sont les approches les plus repandues en production,
    # et leur absence rendait la comparaison incomplete.
    ("A9", "embedding_classifier"),
    ("A10", "lexical_classifier"),
    ("A11", "nearest_neighbour"),
    # Proposition d'origine : similarite maximale seule, coupe par delta cosinus.
    ("A12", "hypothetical_delta_top2"),
    # Compositions : la fonction et ses arguments ne sont plus decides par le
    # meme modele. Le classifieur choisit, DIET extrait.
    ("A13", "classifier_diet"),
    ("A14", "classifier_diet_rules"),
    # Arbitrage par argument, mesure sur la validation, puis apprentissage des
    # arguments enumeres qu'aucune extraction de texte ne peut atteindre.
    ("A15", "classifier_diet_arbitrated"),
    ("A16", "classifier_diet_enum"),
    # Controle : la meme chose sans DIET. L'ecart avec A16 mesure ce que
    # DIET apporte une fois les enumerations traitees.
    ("A17", "classifier_enum"),
)


@dataclass(frozen=True)
class RunSummary:
    architecture: str
    metrics: dict[str, Any]
    environment: dict[str, Any]
    directory: Path

    @property
    def is_publishable(self) -> bool:
        # Un depot modifie rend le run irreproductible : il reste consultable,
        # il n'est pas publie.
        return not self.environment.get("git_dirty", True)

    @property
    def exact_calls(self) -> float | None:
        # Recalcule depuis les predictions : la colonne existe donc aussi pour
        # les runs anterieurs a cette metrique, sans en reecrire aucun.
        return exact_call_rate(self.directory)


def load_summaries() -> dict[str, RunSummary]:
    """Dernier run publiable par architecture."""
    latest: dict[str, RunSummary] = {}
    for directory in iter_runs(complete_only=True):
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        environment = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
        summary = RunSummary(metrics["architecture"], metrics, environment, directory)
        if summary.is_publishable:
            latest[summary.architecture] = summary
    return latest


def _percent(value: float | None) -> str:
    return NOT_RUN if value is None else f"{value:.1%}"


def _milliseconds(value: float | None) -> str:
    return NOT_RUN if value is None else f"{value:.0f} ms"


def render(summaries: dict[str, RunSummary]) -> str:
    """Construit la zone generee, cellule par cellule."""
    lines: list[str] = []

    published = [s for s in summaries.values() if s.metrics.get("count")]
    if not published:
        lines.append(
            "_Aucune campagne publiable n'a encore été exécutée. Cette zone est "
            "générée par `ivr-bench readme update` et ne doit pas être éditée à "
            "la main._\n"
        )
    else:
        # Le run le plus recent sert de reference d'entete. Les lignes viennent
        # de campagnes distinctes : afficher le commit de la premiere laisserait
        # croire que tout le tableau a ete mesure d'un coup.
        reference = max(published, key=lambda s: str(s.environment["timestamp"])).environment
        commits = {s.environment["git_commit"][:12] for s in published}
        lines.append(
            f"_Dernière campagne le {reference['timestamp']} — "
            f"{reference['cpu']}, {reference['cpu_threads']} fils, "
            f"{reference['ram_gb']} Go, accélérateur : {reference['gpu']}. "
            f"Les {len(published)} lignes proviennent de campagnes distinctes, "
            f"réparties sur {len(commits)} commits ; chacune est tracée dans "
            f"`results/runs/`._\n"
        )

    lines.append(
        "| Architecture | Appel exact | Tool accuracy | Macro F1 | Rappel urgence "
        "| Rappel no_tool | Argument EM | Hallucination | p95 | Cas |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for identifier, name in ARCHITECTURES:
        summary = summaries.get(name)
        if summary is None:
            cells = [NOT_RUN] * 8
        else:
            metrics = summary.metrics
            cells = [
                _percent(summary.exact_calls),
                _percent(metrics.get("tool_accuracy")),
                _percent(metrics.get("macro_f1")),
                _percent(metrics.get("emergency_handoff_recall")),
                _percent(metrics.get("no_tool_recall")),
                _percent(metrics.get("argument_exact_match")),
                _percent(metrics.get("hallucinated_argument_rate")),
                _milliseconds((metrics.get("latency_ms") or {}).get("p95")),
            ]
            coverage = metrics.get("coverage", {})
            evaluated = coverage.get("evaluated")
            available = coverage.get("available")
            # Une couverture partielle est annoncee dans la cellule elle-meme :
            # comparer 84 cas a 2 088 sans le dire serait trompeur.
            cells.append(
                f"{evaluated} / {available}"
                if coverage.get("restricted")
                else str(evaluated or NOT_RUN)
            )
        if len(cells) == 8:
            cells.append(NOT_RUN)
        lines.append(f"| {identifier} {name} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "Une cellule `non exécuté` signifie exactement cela : la mesure n'a pas été "
        "faite. Elle ne vaut pas zéro."
    )
    lines.append("")
    lines.append(
        "**Appel exact** : la fonction et *tous* ses arguments sont corrects, compté "
        "sur l'ensemble du corpus. C'est ce qu'un serveur vocal peut exécuter sans "
        "reposer de question. **Argument EM** se compte clé par clé et seulement sur "
        "les cas où la fonction est correcte, donc sur un sous-ensemble différent pour "
        "chaque architecture : les deux colonnes ne classent pas dans le même ordre."
    )

    restricted = [s for s in published if s.metrics.get("coverage", {}).get("restricted")]
    if restricted:
        lines.append("")
        lines.append(
            "Couverture partielle sur : "
            + ", ".join(sorted(s.architecture for s in restricted))
            + ". Ces architectures coûtent plusieurs secondes par énoncé sur CPU ; "
            "l'échantillon est stratifié par fonction et sa taille figure dans la "
            "colonne « Cas »."
        )

    if published:
        lines.append("")
        lines.append("Résultats bruts : `results/runs/`. Reproduction : `make reproduce`.")
    return "\n".join(lines)


def readme_path() -> Path:
    return repo_root() / "README.md"


def update(path: Path | None = None) -> bool:
    """Reecrit la zone generee. Renvoie True si le fichier a change."""
    target = path or readme_path()
    original = target.read_text(encoding="utf-8")
    before, marker, rest = original.partition(START)
    _, end_marker, after = rest.partition(END)
    if not marker or not end_marker:
        raise ValueError(f"marqueurs absents de {target}")

    updated = f"{before}{START}\n\n{render(load_summaries())}\n\n{END}{after}"
    if updated == original:
        return False
    target.write_text(updated, encoding="utf-8")
    return True


def check(path: Path | None = None) -> bool:
    """Verifie que la zone generee est a jour, sans rien ecrire."""
    target = path or readme_path()
    original = target.read_text(encoding="utf-8")
    before, marker, rest = original.partition(START)
    _, end_marker, after = rest.partition(END)
    if not marker or not end_marker:
        raise ValueError(f"marqueurs absents de {target}")
    expected = f"{before}{START}\n\n{render(load_summaries())}\n\n{END}{after}"
    return expected == original
