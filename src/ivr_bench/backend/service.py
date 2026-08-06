"""Operations metier du backend simule.

Ce module est le seul a ecrire. Il applique les invariants de securite qui ne
sont jamais delegues a un modele (§25) : pas de lecture sans patient
authentifie, pas d'ecriture sans confirmation explicite, et jamais d'identifiant
de praticien invente.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from sqlalchemy import select

from ivr_bench.backend.store import (
    OPENING_HOURS,
    Appointment,
    BackendStore,
    is_available,
)
from ivr_bench.domain.clock import reference_now
from ivr_bench.domain.dates import SpokenDateTime

OperationStatus = Literal[
    "ok",
    "authentication_required",
    "practitioner_unknown",
    "no_availability",
    "confirmation_required",
    "not_found",
]


@dataclass(frozen=True)
class AppointmentView:
    """Rendez-vous tel que restitue a l'appelant."""

    appointment_id: str
    practitioner_id: str
    practitioner_name: str
    day: date
    hour: int


@dataclass(frozen=True)
class OperationResult:
    """Resultat d'une operation metier, toujours explicite sur son statut."""

    status: OperationStatus
    appointments: tuple[AppointmentView, ...] = ()
    proposed: tuple[tuple[str, date, int], ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"


class BackendService:
    """Facade metier utilisee par le gestionnaire de dialogue et le benchmark."""

    def __init__(self, store: BackendStore) -> None:
        self._store = store

    # -- lecture ------------------------------------------------------------

    def list_appointments(
        self,
        patient_id: str | None,
        date_from: date | None = None,
        date_to: date | None = None,
        practitioner_id: str | None = None,
    ) -> OperationResult:
        """Consulte les rendez-vous du patient authentifie."""
        # Sans session authentifiee, aucune donnee n'est lue : le patient_id ne
        # peut pas venir de la parole.
        if not patient_id:
            return OperationResult(status="authentication_required")

        statement = select(Appointment).where(Appointment.patient_id == patient_id)
        if date_from is not None:
            statement = statement.where(Appointment.day >= date_from)
        if date_to is not None:
            statement = statement.where(Appointment.day <= date_to)
        if practitioner_id is not None:
            statement = statement.where(Appointment.practitioner_id == practitioner_id)

        with self._store.session() as session:
            rows = session.execute(statement.order_by(Appointment.day, Appointment.hour))
            found = tuple(self._view(row) for row in rows.scalars())
        return OperationResult(status="ok", appointments=found)

    def availability(
        self,
        practitioner_id: str,
        when: SpokenDateTime | None = None,
        limit: int = 3,
    ) -> OperationResult:
        """Propose des creneaux libres pour un praticien."""
        if practitioner_id not in self._store.practitioners:
            return OperationResult(status="practitioner_unknown")

        start, horizon_end = self._store.horizon()
        if when is not None and when.day is not None:
            # Une date precise borne la recherche au jour dit ; une periode
            # (« la semaine prochaine ») la borne a l'intervalle exprime.
            start = max(start, when.day)
            horizon_end = when.range_end if when.range_end is not None else when.day

        booked = self._booked(practitioner_id)
        proposals: list[tuple[str, date, int]] = []
        day = start
        while day <= horizon_end and len(proposals) < limit:
            for hour in OPENING_HOURS:
                if when is not None and when.start_hour is not None:
                    end_hour = when.end_hour if when.end_hour is not None else when.start_hour + 1
                    if not (when.start_hour <= hour < end_hour):
                        continue
                if (day, hour) in booked or not is_available(practitioner_id, day, hour):
                    continue
                proposals.append((practitioner_id, day, hour))
                if len(proposals) >= limit:
                    break
            day += timedelta(days=1)

        if not proposals:
            return OperationResult(status="no_availability")
        return OperationResult(status="ok", proposed=tuple(proposals))

    # -- ecriture -----------------------------------------------------------

    def create_appointment(
        self,
        patient_id: str | None,
        practitioner_id: str,
        day: date,
        hour: int,
        confirmed: bool,
        dry_run: bool = False,
    ) -> OperationResult:
        """Cree un rendez-vous, uniquement apres confirmation explicite."""
        if not patient_id:
            return OperationResult(status="authentication_required")
        if practitioner_id not in self._store.practitioners:
            return OperationResult(status="practitioner_unknown")
        # La confirmation n'est pas une formalite : c'est l'invariant qui empeche
        # un modele de creer un rendez-vous a partir d'une phrase mal comprise.
        if not confirmed:
            return OperationResult(status="confirmation_required")
        if (day, hour) in self._booked(practitioner_id) or not is_available(
            practitioner_id, day, hour
        ):
            return OperationResult(status="no_availability")

        appointment_id = self._identifier(patient_id, practitioner_id, day, hour)
        if dry_run:
            return OperationResult(status="ok", detail={"dry_run": True, "id": appointment_id})

        with self._store.session() as session:
            session.add(
                Appointment(
                    appointment_id=appointment_id,
                    patient_id=patient_id,
                    practitioner_id=practitioner_id,
                    day=day,
                    hour=hour,
                    created_at=reference_now().replace(tzinfo=None),
                )
            )
            session.commit()
        return OperationResult(status="ok", detail={"id": appointment_id})

    def reschedule_appointment(
        self,
        patient_id: str | None,
        appointment_id: str,
        day: date,
        hour: int,
        confirmed: bool,
        dry_run: bool = False,
    ) -> OperationResult:
        """Deplace un rendez-vous existant du patient authentifie."""
        if not patient_id:
            return OperationResult(status="authentication_required")
        if not confirmed:
            return OperationResult(status="confirmation_required")

        with self._store.session() as session:
            appointment = session.get(Appointment, appointment_id)
            # Un patient ne deplace que ses propres rendez-vous : l'appartenance
            # est verifiee ici, pas deduite de la demande.
            if appointment is None or appointment.patient_id != patient_id:
                return OperationResult(status="not_found")

            practitioner_id = appointment.practitioner_id
            if (day, hour) in self._booked(practitioner_id, exclude=appointment_id) or (
                not is_available(practitioner_id, day, hour)
            ):
                return OperationResult(status="no_availability")

            if dry_run:
                return OperationResult(status="ok", detail={"dry_run": True, "id": appointment_id})

            appointment.day = day
            appointment.hour = hour
            session.commit()
        return OperationResult(status="ok", detail={"id": appointment_id})

    # -- interne ------------------------------------------------------------

    def _booked(self, practitioner_id: str, exclude: str | None = None) -> set[tuple[date, int]]:
        statement = select(Appointment).where(Appointment.practitioner_id == practitioner_id)
        with self._store.session() as session:
            rows = session.execute(statement).scalars()
            return {
                (row.day, row.hour)
                for row in rows
                if exclude is None or row.appointment_id != exclude
            }

    def _view(self, appointment: Appointment) -> AppointmentView:
        practitioner = self._store.practitioners.get(appointment.practitioner_id)
        return AppointmentView(
            appointment_id=appointment.appointment_id,
            practitioner_id=appointment.practitioner_id,
            practitioner_name=practitioner.display_name if practitioner else "praticien inconnu",
            day=appointment.day,
            hour=appointment.hour,
        )

    @staticmethod
    def _identifier(patient_id: str, practitioner_id: str, day: date, hour: int) -> str:
        """Identifiant opaque et stable, sans donnee personnelle en clair."""
        digest = hashlib.sha256(
            f"{patient_id}|{practitioner_id}|{day.isoformat()}|{hour}".encode()
        ).hexdigest()
        return f"appointment_{digest[:12]}"
