"""Localisation des ressources du depot.

Aucun chemin absolu n'est code en dur (§33). La racine est deduite d'un marqueur
present dans le depot, ou imposee par la variable d'environnement
`IVR_BENCH_ROOT` lorsque le paquet est installe ailleurs que dans ses sources.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Marqueur suffisamment specifique pour ne pas confondre le depot avec un parent.
_MARKER = Path("config") / "domain" / "functions.yaml"


def _search_upwards(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / _MARKER).is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Racine du depot."""
    override = os.environ.get("IVR_BENCH_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / _MARKER).is_file():
            raise FileNotFoundError(f"IVR_BENCH_ROOT={root} ne contient pas {_MARKER}.")
        return root

    for start in (Path.cwd(), Path(__file__).resolve().parent):
        found = _search_upwards(start.resolve())
        if found is not None:
            return found

    raise FileNotFoundError(
        "Racine du depot introuvable : definissez IVR_BENCH_ROOT vers le depot cloné."
    )


def config_dir() -> Path:
    return repo_root() / "config"


def data_dir() -> Path:
    return repo_root() / "data"


def results_dir() -> Path:
    return repo_root() / "results"


def reports_dir() -> Path:
    return repo_root() / "reports"


def weights_dir() -> Path:
    """Cache des poids reels, hors du depot et jamais versionne."""
    override = os.environ.get("IVR_BENCH_WEIGHTS_DIR")
    return Path(override).expanduser() if override else repo_root() / "weights"
