"""Tests de l'interface en ligne de commande."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ivr_bench import __version__
from ivr_bench.cli.main import app

runner = CliRunner()

# Commandes imposees par le paragraphe 20 de la specification.
EXPECTED_COMMANDS = [
    ["data", "generate"],
    ["data", "validate"],
    ["doctors", "generate"],
    ["diet", "train"],
    ["index", "build"],
    ["benchmark", "text"],
    ["benchmark", "audio"],
    ["benchmark", "e2e"],
    ["benchmark", "all"],
    ["report", "build"],
    ["readme", "update"],
    ["reproduce"],
]


def test_version_matches_package() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


@pytest.mark.parametrize("command", EXPECTED_COMMANDS, ids=lambda c: " ".join(c))
def test_command_is_registered(command: list[str]) -> None:
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, f"commande absente : {' '.join(command)}"


@pytest.mark.parametrize("command", EXPECTED_COMMANDS, ids=lambda c: " ".join(c))
def test_unimplemented_command_fails_loudly(command: list[str]) -> None:
    """Une etape non implementee echoue : elle ne renvoie jamais de faux resultat."""
    result = runner.invoke(app, command)
    assert result.exit_code != 0
