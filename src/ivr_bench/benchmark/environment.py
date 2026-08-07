"""Contexte d'execution enregistre avec chaque campagne (§28).

Un resultat sans son materiel, ses versions et son commit n'est pas un resultat :
il n'est pas rejouable, donc pas comparable. Un run execute avec un depot modifie
non commite est marque `dirty` et ne remplacera jamais les chiffres publies.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from ivr_bench.domain.paths import repo_root


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def working_tree_is_modified() -> bool:
    """Le depot porte-t-il des modifications non commitees ?

    Les resultats sont exclus du constat. Depuis qu'ils sont versionnes, une
    campagne salit l'arbre en ecrivant son propre run : sans cette exclusion,
    la premiere campagne d'une serie rendrait toutes les suivantes
    « irreproductibles » alors que rien du code mesure n'aurait bouge. Ce que le
    drapeau doit dire est precis — le code execute n'est pas celui du commit —
    et un fichier de resultats ne change pas le code execute.
    """
    return bool(_git("status", "--porcelain", "--", ":!results/runs"))


def _dependency_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    names = (
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers",
        "jax",
        "rapidfuzz",
        "pydantic",
    )
    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            # Un paquet absent est declare comme tel : il fait partie du contexte.
            found[name] = "absent"
    return found


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunEnvironment:
    """Tout ce qu'exige le §28, plus rien d'invente."""

    run_id: str
    timestamp: str
    git_commit: str
    git_dirty: bool
    os: str
    kernel: str
    python: str
    dependencies: dict[str, str]
    cpu: str
    cpu_threads: int
    ram_gb: float
    # Aucun accelerateur : c'est la premisse de l'experience, pas une mesure
    # manquante. Les colonnes GPU du §17.5 sont donc non applicables.
    gpu: str
    runtime: str
    seeds: list[int]
    dataset_hashes: dict[str, str]
    config_hash: str
    command: str
    duration_s: float = 0.0
    models: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture(
    run_id: str,
    seeds: list[int],
    command: str,
    config_path: Path | None = None,
    dataset_paths: dict[str, Path] | None = None,
    models: dict[str, str] | None = None,
) -> RunEnvironment:
    """Photographie l'environnement au demarrage d'une campagne."""
    dirty = working_tree_is_modified()
    config_hash = file_digest(config_path) if config_path and config_path.is_file() else ""

    hashes: dict[str, str] = {}
    for label, path in (dataset_paths or {}).items():
        if path.is_file():
            hashes[label] = file_digest(path)

    return RunEnvironment(
        run_id=run_id,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=dirty,
        os=platform.platform(),
        kernel=platform.release(),
        python=sys.version.split()[0],
        dependencies=_dependency_versions(),
        cpu=platform.processor() or platform.machine(),
        cpu_threads=psutil.cpu_count(logical=True) or 0,
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        gpu="aucun (comparaison CPU par construction)",
        runtime="cpu",
        seeds=seeds,
        dataset_hashes=hashes,
        config_hash=config_hash,
        command=command,
        models=models or {},
    )


def write(environment: RunEnvironment, directory: Path, duration_s: float) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = environment.to_dict()
    payload["duration_s"] = round(duration_s, 2)
    target = directory / "environment.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
