"""Tests du catalogue metier canonique."""

from __future__ import annotations

import pytest

from ivr_bench.domain.catalog import (
    default_catalog,
    load_general_information,
    load_safety_policy,
)

EXPECTED_FUNCTIONS = {
    "answer_general_information",
    "list_appointments",
    "request_new_appointment",
    "request_appointment_reschedule",
    "emergency_handoff",
    "human_handoff",
    "no_tool",
}


def test_catalog_contains_the_specified_functions() -> None:
    assert set(default_catalog().names) == EXPECTED_FUNCTIONS


def test_no_tool_is_never_executable() -> None:
    """La pseudo-fonction ne doit jamais atteindre le backend metier (§5.7)."""
    catalog = default_catalog()
    assert catalog.get("no_tool").executable is False
    assert "no_tool" not in catalog.executable_names


def test_patient_id_is_session_injected_and_absent_from_every_signature() -> None:
    """Le patient_id vient de la session, jamais de la parole (§5.2, §25.3)."""
    catalog = default_catalog()
    assert "patient_id" in catalog.session_injected
    for definition in catalog.functions:
        assert "patient_id" not in definition.parameter_names


def test_no_function_exposes_a_practitioner_identifier() -> None:
    """Le modele transmet un nom prononce ; l'identite releve du resolveur (§8)."""
    for definition in default_catalog().functions:
        assert "practitioner_id" not in definition.parameter_names


def test_appointment_reason_is_never_persisted() -> None:
    """Le motif peut relever du soin : il n'entre pas dans les resultats (§5.3)."""
    reason = default_catalog().get("request_new_appointment").parameter("reason")
    assert reason is not None
    assert reason.persisted is False
    assert "reason" not in default_catalog().get("request_new_appointment").persisted_parameters


def test_appointment_arguments_are_all_optional() -> None:
    """Une demande peut selectionner la bonne fonction sans fournir d'argument (§12)."""
    definition = default_catalog().get("request_new_appointment")
    assert definition.required_parameters == ()


def test_enumerated_arguments_are_required() -> None:
    catalog = default_catalog()
    assert catalog.get("emergency_handoff").required_parameters == ("reason_category",)
    assert catalog.get("human_handoff").required_parameters == ("reason",)
    assert catalog.get("answer_general_information").required_parameters == ("topic",)


def test_every_function_carries_generation_seeds() -> None:
    """Les amorces alimentent la generation offline (§10.1)."""
    for definition in default_catalog().functions:
        assert definition.positive_seeds, f"{definition.name} n'a aucune amorce"
        assert definition.safety_notes, f"{definition.name} n'a aucune note de securite"


def test_confusable_references_point_to_known_functions() -> None:
    catalog = default_catalog()
    for definition in catalog.functions:
        for other in definition.confusable_with:
            assert other in catalog.names


def test_unknown_function_lookup_fails() -> None:
    with pytest.raises(KeyError):
        default_catalog().get("request_appointment_cancellation")


def test_subset_preserves_requested_order() -> None:
    """Le sous-catalogue est le coeur des architectures hybrides."""
    subset = default_catalog().subset(["request_new_appointment", "list_appointments"])
    assert [definition.name for definition in subset] == [
        "request_new_appointment",
        "list_appointments",
    ]


def test_safety_policy_forbids_model_authority() -> None:
    invariants = load_safety_policy()["invariants"]
    assert invariants["allow_patient_id_from_speech"] is False
    assert invariants["allow_model_generated_practitioner_id"] is False
    assert invariants["allow_medical_advice"] is False
    assert invariants["require_explicit_confirmation_before_write"] is True


def test_safety_policy_does_not_log_raw_audio_or_reason() -> None:
    logging_policy = load_safety_policy()["logging"]
    assert logging_policy["store_raw_audio"] is False
    assert logging_policy["store_reason_field"] is False
    assert logging_policy["pseudonymize_identifiers"] is True


def test_general_information_covers_every_topic_of_the_enum() -> None:
    topic = default_catalog().get("answer_general_information").parameter("topic")
    assert topic is not None and topic.enum is not None
    topics = load_general_information()["topics"]
    assert set(topic.enum) == set(topics)
