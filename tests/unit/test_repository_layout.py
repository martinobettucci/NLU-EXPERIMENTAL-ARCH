"""Verifie que la structure imposee par la specification est presente.

Ces tests ne valident pas du code metier : ils empechent qu'un fichier structurant
disparaisse silencieusement d'une campagne a l'autre.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "SPECIFICATION_IVR_ROUTING_BENCHMARK.md",
    "architecture.md",
    "dataset.md",
    "benchmark_protocol.md",
    "safety.md",
    "reproducibility.md",
    "adding_an_architecture.md",
    "adding_a_function.md",
    "results_interpretation.md",
    "references.md",
]

REQUIRED_PATHS = [
    "Makefile",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "runDev",
    "runBenchmark",
    "runProd",
    "docker-compose.dev.yml",
    "docker-compose.benchmark.yml",
    "docker-compose.prod.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/benchmark-smoke.yml",
    ".github/workflows/benchmark-full.yml",
    ".github/workflows/publish-results.yml",
]

REQUIRED_PACKAGES = [
    "domain",
    "generators",
    "routers",
    "embeddings",
    "retrieval",
    "resolver",
    "dialogue",
    "asr",
    "tts",
    "backend",
    "benchmark",
    "metrics",
    "reporting",
    "api",
    "cli",
]


@pytest.mark.parametrize("relative", REQUIRED_PATHS)
def test_structural_file_exists(relative: str) -> None:
    assert (REPO_ROOT / relative).is_file(), f"fichier structurant absent : {relative}"


@pytest.mark.parametrize("name", REQUIRED_DOCS)
def test_documentation_exists(name: str) -> None:
    document = REPO_ROOT / "docs" / name
    assert document.is_file(), f"document absent : docs/{name}"
    assert document.stat().st_size > 0, f"document vide : docs/{name}"


@pytest.mark.parametrize("name", REQUIRED_PACKAGES)
def test_package_module_exists(name: str) -> None:
    assert (REPO_ROOT / "src" / "ivr_bench" / name / "__init__.py").is_file()


@pytest.mark.parametrize("script", ["runDev", "runBenchmark", "runProd"])
def test_run_scripts_are_executable(script: str) -> None:
    import os

    assert os.access(REPO_ROOT / script, os.X_OK), f"{script} n'est pas executable"
