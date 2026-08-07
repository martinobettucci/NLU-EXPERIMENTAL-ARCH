"""Taux d'appel exact : ce que la métrique compte, et ce qu'elle refuse.

Elle existe parce que l'exactitude d'arguments récompense la justesse
partielle. Ces tests fixent la frontière : un appel dont un seul argument
diverge n'est pas un appel exact, même si tous les autres sont bons.
"""

from __future__ import annotations

import json
from pathlib import Path

from ivr_bench.metrics.calls import exact_call_rate, is_exact_call


def test_identical_call_is_exact() -> None:
    call = {"tool_name": "list_appointments", "arguments": {"date_from": "mardi"}}
    assert is_exact_call(call, dict(call))


def test_a_single_wrong_argument_loses_the_call() -> None:
    expected = {
        "tool_name": "request_new_appointment",
        "arguments": {"practitioner_name": "Rey", "preferred_date": "mardi"},
    }
    produced = {
        "tool_name": "request_new_appointment",
        "arguments": {"practitioner_name": "Rey", "preferred_date": "vendredi"},
    }
    assert not is_exact_call(expected, produced)


def test_case_and_spacing_do_not_count_as_a_difference() -> None:
    expected = {"tool_name": "list_appointments", "arguments": {"practitioner_name": "Rey"}}
    produced = {"tool_name": "list_appointments", "arguments": {"practitioner_name": "  REY "}}
    assert is_exact_call(expected, produced)


def test_an_extra_argument_loses_the_call() -> None:
    """Remplir un champ que l'énoncé n'exprime pas reste une erreur (§12)."""
    expected = {"tool_name": "list_appointments", "arguments": {"date_from": None}}
    produced = {"tool_name": "list_appointments", "arguments": {"date_from": "demain"}}
    assert not is_exact_call(expected, produced)


def test_a_wrong_function_loses_the_call() -> None:
    expected = {"tool_name": "list_appointments", "arguments": {}}
    produced = {"tool_name": "request_new_appointment", "arguments": {}}
    assert not is_exact_call(expected, produced)


def test_rate_is_computed_over_every_case(tmp_path: Path) -> None:
    rows = [
        {
            "expected": {"tool_name": "no_tool", "arguments": {}},
            "predicted": {"tool_name": "no_tool", "arguments": {}},
        },
        {
            "expected": {"tool_name": "no_tool", "arguments": {}},
            "predicted": {"tool_name": "human_handoff", "arguments": {}},
        },
    ]
    (tmp_path / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    assert exact_call_rate(tmp_path) == 0.5


def test_a_run_without_predictions_reports_nothing(tmp_path: Path) -> None:
    """Absence de mesure, pas zéro : la distinction est la règle du dépôt."""
    assert exact_call_rate(tmp_path) is None
