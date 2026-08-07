"""Tests de l'interface en ligne de commande."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ivr_bench import __version__
from ivr_bench.cli.main import app

runner = CliRunner()

# Commandes du paragraphe 20 dont l'etape n'est pas encore implementee : elles
# doivent echouer franchement. Une commande implementee ne figure pas ici, sinon
# le test la lancerait pour de bon — « benchmark text » declencherait une
# campagne complete au milieu de la suite unitaire.
EXPECTED_COMMANDS = [
    ["diet", "train"],
    ["benchmark", "audio"],
    ["benchmark", "e2e"],
    ["report", "build"],
    ["reproduce"],
]


# Commandes deja implementees : elles doivent apparaitre dans l'arbre et
# repondre, pas echouer.
IMPLEMENTED_COMMANDS = [
    ["benchmark", "text"],
    ["benchmark", "all"],
    ["readme", "update"],
    ["readme", "check"],
    ["doctors", "generate"],
    ["index", "build"],
    ["models", "download"],
    ["data", "generate"],
    ["data", "validate"],
    ["domain", "show"],
    ["domain", "schemas"],
]


def test_version_matches_package() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


@pytest.mark.parametrize(
    "command", [*EXPECTED_COMMANDS, *IMPLEMENTED_COMMANDS], ids=lambda c: " ".join(c)
)
def test_command_is_registered(command: list[str]) -> None:
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, f"commande absente : {' '.join(command)}"


@pytest.mark.parametrize("command", EXPECTED_COMMANDS, ids=lambda c: " ".join(c))
def test_unimplemented_command_fails_loudly(command: list[str]) -> None:
    """Une etape non implementee echoue : elle ne renvoie jamais de faux resultat."""
    result = runner.invoke(app, command)
    assert result.exit_code != 0
