"""Tests du backend metier simule (§6) et de ses invariants de securite (§25)."""

from __future__ import annotations

from datetime import date

import pytest

from ivr_bench.backend.service import AppointmentView, BackendService
from ivr_bench.backend.store import BackendStore, is_available
from ivr_bench.domain.dates import parse
from ivr_bench.domain.models import Practitioner
from ivr_bench.generators.practitioners import generate_practitioners

PATIENT = "patient_00001"


@pytest.fixture
def practitioners() -> list[Practitioner]:
    return generate_practitioners(seed=42, count=120)


@pytest.fixture
def service(practitioners: list[Practitioner]) -> BackendService:
    store = BackendStore(practitioners, url="sqlite+pysqlite:///:memory:")
    store.seed(seed=42, patient_count=20)
    return BackendService(store)


def test_availability_is_deterministic(practitioners: list[Practitioner]) -> None:
    """Le meme praticien a toujours les memes creneaux le meme jour."""
    identifier = practitioners[0].practitioner_id
    day = date(2026, 3, 3)
    first = [hour for hour in range(8, 19) if is_available(identifier, day, hour)]
    second = [hour for hour in range(8, 19) if is_available(identifier, day, hour)]
    assert first == second
    assert first, "aucun creneau ouvert : le backend serait inutilisable"


def test_weekend_is_closed(practitioners: list[Practitioner]) -> None:
    saturday = date(2026, 3, 7)
    assert all(not is_available(practitioners[0].practitioner_id, saturday, h) for h in range(24))


# -- lecture ----------------------------------------------------------------


def test_reading_requires_an_authenticated_session(service: BackendService) -> None:
    """Aucune consultation sans contexte patient : le patient_id vient de la session."""
    assert service.list_appointments(None).status == "authentication_required"
    assert service.list_appointments("").status == "authentication_required"


def test_patient_sees_only_their_own_appointments(service: BackendService) -> None:
    result = service.list_appointments(PATIENT)
    assert result.succeeded
    other = service.list_appointments("patient_00002")
    identifiers = {item.appointment_id for item in result.appointments}
    assert identifiers.isdisjoint({item.appointment_id for item in other.appointments})


def test_period_filter_narrows_the_result(service: BackendService) -> None:
    everything = service.list_appointments(PATIENT)
    narrowed = service.list_appointments(PATIENT, date_from=date(2030, 1, 1))
    assert len(narrowed.appointments) <= len(everything.appointments)
    assert narrowed.appointments == ()


# -- ecriture ---------------------------------------------------------------


def test_creation_requires_explicit_confirmation(
    service: BackendService, practitioners: list[Practitioner]
) -> None:
    """L'invariant qui empeche une phrase mal comprise de creer un rendez-vous."""
    identifier = practitioners[0].practitioner_id
    slot = service.availability(identifier).proposed[0]
    result = service.create_appointment(PATIENT, identifier, slot[1], slot[2], confirmed=False)
    assert result.status == "confirmation_required"


def test_creation_requires_authentication(
    service: BackendService, practitioners: list[Practitioner]
) -> None:
    identifier = practitioners[0].practitioner_id
    slot = service.availability(identifier).proposed[0]
    result = service.create_appointment(None, identifier, slot[1], slot[2], confirmed=True)
    assert result.status == "authentication_required"


def test_unknown_practitioner_is_rejected(service: BackendService) -> None:
    """Un identifiant invente par un modele ne cree rien."""
    result = service.create_appointment(
        PATIENT, "practitioner_99999", date(2026, 3, 3), 9, confirmed=True
    )
    assert result.status == "practitioner_unknown"


def test_confirmed_creation_succeeds_and_takes_the_slot(
    service: BackendService, practitioners: list[Practitioner]
) -> None:
    identifier = practitioners[0].practitioner_id
    _, day, hour = service.availability(identifier).proposed[0]

    created = service.create_appointment(PATIENT, identifier, day, hour, confirmed=True)
    assert created.succeeded

    again = service.create_appointment("patient_00002", identifier, day, hour, confirmed=True)
    assert again.status == "no_availability"


def test_dry_run_changes_nothing(
    service: BackendService, practitioners: list[Practitioner]
) -> None:
    """Une suite ne doit pas modifier l'etat lu par la suivante (§6)."""
    identifier = practitioners[0].practitioner_id
    _, day, hour = service.availability(identifier).proposed[0]

    before = len(service.list_appointments(PATIENT).appointments)
    result = service.create_appointment(
        PATIENT, identifier, day, hour, confirmed=True, dry_run=True
    )
    assert result.succeeded
    assert result.detail["dry_run"] is True
    assert len(service.list_appointments(PATIENT).appointments) == before


def test_reschedule_only_touches_your_own_appointment(service: BackendService) -> None:
    holder, appointment = _first_appointment(service)
    del holder
    # Un autre patient ne peut pas deplacer ce rendez-vous, meme en connaissant
    # son identifiant.
    result = service.reschedule_appointment(
        "patient_00019", appointment.appointment_id, date(2026, 3, 10), 9, confirmed=True
    )
    assert result.status == "not_found"


def test_reschedule_moves_the_appointment(service: BackendService) -> None:
    holder, appointment = _first_appointment(service)
    target = next(
        (day, hour)
        for day, hour in _slots(appointment.practitioner_id)
        if (day, hour) != (appointment.day, appointment.hour)
    )
    result = service.reschedule_appointment(
        holder, appointment.appointment_id, target[0], target[1], confirmed=True
    )
    assert result.succeeded

    moved = next(
        item
        for item in service.list_appointments(holder).appointments
        if item.appointment_id == appointment.appointment_id
    )
    assert (moved.day, moved.hour) == target


def test_availability_honours_a_spoken_time_window(
    service: BackendService, practitioners: list[Practitioner]
) -> None:
    identifier = practitioners[0].practitioner_id
    result = service.availability(identifier, when=parse("mardi matin", reference=date(2026, 3, 2)))
    if result.succeeded:
        for _, day, hour in result.proposed:
            assert day == date(2026, 3, 3)
            assert 8 <= hour < 12


def test_reset_restores_a_clean_state(practitioners: list[Practitioner]) -> None:
    store = BackendStore(practitioners, url="sqlite+pysqlite:///:memory:")
    store.seed(seed=42, patient_count=20)
    service = BackendService(store)

    identifier = practitioners[0].practitioner_id
    _, day, hour = service.availability(identifier).proposed[0]
    service.create_appointment(PATIENT, identifier, day, hour, confirmed=True)
    after_write = len(service.list_appointments(PATIENT).appointments)

    store.reset(seed=42, patient_count=20)
    assert len(service.list_appointments(PATIENT).appointments) == after_write - 1


def _first_appointment(service: BackendService) -> tuple[str, AppointmentView]:
    """Premier patient possedant reellement un rendez-vous.

    Le peuplement laisse volontairement certains patients sans aucun rendez-vous :
    c'est un cas de test a part entiere, pas un patient a corriger.
    """
    for index in range(1, 21):
        patient_id = f"patient_{index:05d}"
        found = service.list_appointments(patient_id).appointments
        if found:
            return patient_id, found[0]
    raise AssertionError("aucun patient ne possede de rendez-vous")


def _slots(practitioner_id: str) -> list[tuple[date, int]]:
    from datetime import timedelta

    start = date(2026, 3, 2)
    return [
        (start + timedelta(days=offset), hour)
        for offset in range(30)
        for hour in range(8, 19)
        if is_available(practitioner_id, start + timedelta(days=offset), hour)
    ]
