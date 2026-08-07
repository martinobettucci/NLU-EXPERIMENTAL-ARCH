"""Tests de l'interface en ligne de commande."""

from __future__ import annotations

import inspect

import click
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from ivr_bench import __version__
from ivr_bench.cli.main import app

runner = CliRunner()

# Commandes imposees par le paragraphe 20 de la specification. Elles doivent
# toutes exister, implementees ou non.
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


def _walk(command: click.Command, prefix: list[str]) -> dict[str, click.Command]:
    """Toutes les commandes terminales de l'arbre, par chemin.

    Le parcours reconnait un groupe a son attribut `commands` plutot qu'a son
    type : selon la version, Typer renvoie une classe qui ne descend pas de
    `click.Group` et le test ne verrait alors qu'une seule feuille vide.
    """
    children = getattr(command, "commands", None)
    if children:
        found: dict[str, click.Command] = {}
        for name, child in children.items():
            found.update(_walk(child, [*prefix, name]))
        return found
    return {" ".join(prefix): command}


def _leaf_commands() -> dict[str, click.Command]:
    return _walk(get_command(app), [])


def _is_pending(command: click.Command) -> bool:
    """Une commande est en attente si son corps appelle `_pending`.

    Le critere est lu dans le code plutot que maintenu dans une liste. Une liste
    ecrite a la main devient fausse des qu'une etape est implementee, et le test
    se met alors a **executer** pour de bon la commande qu'il croyait inerte —
    ce qui a deja declenche une campagne complete au milieu de la suite
    unitaire, puis la chaine de reproduction entiere.
    """
    callback = command.callback
    if callback is None:
        return False
    try:
        return "_pending(" in inspect.getsource(callback)
    except OSError:  # pragma: no cover - source indisponible
        return False


def test_version_matches_package() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


@pytest.mark.parametrize("command", EXPECTED_COMMANDS, ids=lambda c: " ".join(c))
def test_specified_command_exists(command: list[str]) -> None:
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, f"commande absente : {' '.join(command)}"


def test_every_command_answers_to_help() -> None:
    for path in _leaf_commands():
        result = runner.invoke(app, [*path.split(" "), "--help"])
        assert result.exit_code == 0, path


def test_pending_commands_fail_loudly() -> None:
    """Une etape non implementee echoue : elle ne renvoie jamais de faux resultat.

    Seules les commandes reellement en attente sont invoquees, ce qui garantit
    qu'aucune commande implementee n'est declenchee par ce test.
    """
    pending = {path for path, command in _leaf_commands().items() if _is_pending(command)}
    assert pending, "aucune commande en attente : la detection est cassee"

    for path in sorted(pending):
        result = runner.invoke(app, path.split(" "))
        assert result.exit_code != 0, path
