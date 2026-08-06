"""Garde-fous sur la zone generee du README.

La specification impose que seule la zone entre marqueurs soit reecrite, et
qu'une metrique non mesuree affiche 'non execute' plutot que zero. Ces tests
verrouillent ces deux proprietes des maintenant, avant meme que le generateur
de rapports existe.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
START = "<!-- BENCHMARK_RESULTS_START -->"
END = "<!-- BENCHMARK_RESULTS_END -->"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _generated_zone(text: str) -> str:
    return text.split(START, 1)[1].split(END, 1)[0]


def test_markers_are_present_exactly_once(readme_text: str) -> None:
    assert readme_text.count(START) == 1
    assert readme_text.count(END) == 1


def test_markers_are_ordered(readme_text: str) -> None:
    assert readme_text.index(START) < readme_text.index(END)


def test_generated_zone_reports_unmeasured_cells_explicitly(readme_text: str) -> None:
    """Une cellule sans mesure vaut 'non execute', jamais un nombre."""
    zone = _generated_zone(readme_text)
    # Les lignes d'architecture commencent par 'A' suivi de son numero : cela exclut
    # l'en-tete 'Architecture'.
    rows = [line for line in zone.splitlines() if re.match(r"\| A\d ", line)]
    assert rows, "le tableau principal doit lister les architectures"

    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")[1:]]
        for cell in cells:
            if cell == "non exécuté":
                continue
            assert re.fullmatch(r"[\d.,]+\s*\S*", cell), (
                f"cellule inattendue dans la zone generee : {cell!r}. "
                "Une metrique absente doit afficher 'non exécuté'."
            )


def test_all_nine_architectures_are_listed(readme_text: str) -> None:
    zone = _generated_zone(readme_text)
    for architecture in (
        "rules",
        "diet",
        "needle_full",
        "functiongemma_zero_shot",
        "functiongemma_tuned",
        "embedding_only",
        "hybrid_needle_top2",
        "hybrid_functiongemma_top2",
        "hybrid_adaptive",
    ):
        assert architecture in zone, f"architecture absente du tableau : {architecture}"
