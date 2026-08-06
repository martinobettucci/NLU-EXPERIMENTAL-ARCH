"""Tests de la chaine de validation des sorties de modeles (§24)."""

from __future__ import annotations

import pytest

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.validation import OutputValidator


@pytest.fixture(scope="module")
def validator() -> OutputValidator:
    return OutputValidator(default_catalog())


def test_clean_output_is_native(validator: OutputValidator) -> None:
    outcome = validator.validate(
        '{"name": "request_new_appointment", "arguments": '
        '{"practitioner_name": "docteur Bensaid", "preferred_date": "mardi"}}'
    )
    assert outcome.validity == "native"
    assert outcome.tool_name == "request_new_appointment"
    assert outcome.arguments["preferred_date"] == "mardi"
    assert outcome.repairs == ()


def test_missing_arguments_are_null_not_invented(validator: OutputValidator) -> None:
    """Une demande sans argument reste valide : le routeur n'invente rien (§12)."""
    outcome = validator.validate(
        '{"name": "request_new_appointment", "arguments": '
        '{"practitioner_name": null, "specialty": null, "preferred_date": null, '
        '"preferred_time": null, "reason": null}}'
    )
    assert outcome.validity == "native"
    assert outcome.arguments["practitioner_name"] is None


def test_code_fence_is_a_repair_not_a_free_pass(validator: OutputValidator) -> None:
    """La sortie reste utilisable, mais comptabilisee a part des resultats natifs."""
    outcome = validator.validate('```json\n{"name": "list_appointments", "arguments": {}}\n```')
    assert outcome.validity == "repaired"
    assert "bloc de code retire" in outcome.repairs


@pytest.mark.parametrize(
    "raw",
    [
        '{"tool_name": "list_appointments", "arguments": {}}',
        '{"name": "list_appointments", "parameters": {}}',
        '{"name": "list_appointments", "arguments": "{}"}',
        '[{"name": "list_appointments", "arguments": {}}]',
    ],
)
def test_alternative_conventions_are_repaired(validator: OutputValidator, raw: str) -> None:
    outcome = validator.validate(raw)
    assert outcome.validity == "repaired"
    assert outcome.tool_name == "list_appointments"


def test_unparsable_output_is_invalid(validator: OutputValidator) -> None:
    assert validator.validate("je vais chercher vos rendez-vous").validity == "invalid"


def test_unknown_function_is_invalid(validator: OutputValidator) -> None:
    outcome = validator.validate('{"name": "cancel_appointment", "arguments": {}}')
    assert outcome.validity == "invalid"
    assert "hors catalogue" in outcome.errors[0]


def test_hallucinated_argument_is_invalid(validator: OutputValidator) -> None:
    """Un argument absent du schema est une erreur, pas une donnee bonus."""
    outcome = validator.validate(
        '{"name": "list_appointments", "arguments": {"insurance_number": "12345"}}'
    )
    assert outcome.validity == "invalid"


def test_model_may_not_produce_a_patient_identifier(validator: OutputValidator) -> None:
    outcome = validator.validate(
        '{"name": "list_appointments", "arguments": {"patient_id": "patient_0001"}}'
    )
    assert outcome.validity == "invalid"
    assert any("interdits" in error for error in outcome.errors)


def test_model_may_not_produce_a_practitioner_identifier(validator: OutputValidator) -> None:
    outcome = validator.validate(
        '{"name": "list_appointments", "arguments": {"practitioner_id": "practitioner_00421"}}'
    )
    assert outcome.validity == "invalid"
    assert any("interdits" in error for error in outcome.errors)


def test_value_outside_enumeration_is_invalid(validator: OutputValidator) -> None:
    outcome = validator.validate(
        '{"name": "answer_general_information", "arguments": {"topic": "meteo"}}'
    )
    assert outcome.validity == "invalid"


def test_required_argument_missing_is_invalid(validator: OutputValidator) -> None:
    outcome = validator.validate('{"name": "emergency_handoff", "arguments": {}}')
    assert outcome.validity == "invalid"


def test_no_tool_takes_no_argument(validator: OutputValidator) -> None:
    assert validator.validate('{"name": "no_tool", "arguments": {}}').validity == "native"
    assert (
        validator.validate('{"name": "no_tool", "arguments": {"topic": "other"}}').validity
        == "invalid"
    )


def test_multiple_calls_are_not_silently_truncated(validator: OutputValidator) -> None:
    """Deux appels ne sont pas la reponse attendue : on ne garde pas le premier."""
    outcome = validator.validate(
        '[{"name": "list_appointments", "arguments": {}}, {"name": "no_tool", "arguments": {}}]'
    )
    assert outcome.validity == "invalid"


def test_unpersisted_arguments_are_stripped_from_results(validator: OutputValidator) -> None:
    kept = validator.strip_unpersisted(
        "request_new_appointment",
        {"practitioner_name": "docteur Rey", "reason": "douleur au genou"},
    )
    assert kept == {"practitioner_name": "docteur Rey"}
