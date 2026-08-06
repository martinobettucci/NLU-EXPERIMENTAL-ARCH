"""Backend metier simule (§6).

Entierement synthetique, auto-initialise, reproductible. Deux choix structurants :

1. **Les disponibilites ne sont pas stockees.** Elles sont derivees d'une
   empreinte stable de (praticien, date), donc identiques d'une execution a
   l'autre sans peupler des centaines de milliers de creneaux. Seuls les
   rendez-vous, qui sont des faits, vivent en base.
2. **L'horloge est figee.** Les dates relatives du corpus tombent toujours au
   meme endroit du calendrier, sinon les reponses attendues changeraient chaque
   jour et les campagnes cesseraient d'etre comparables.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import Date, DateTime, ForeignKey, String, create_engine, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from ivr_bench.domain.clock import reference_now
from ivr_bench.domain.models import Practitioner

# Heures ouvrables du centre, sur lesquelles les creneaux sont derives.
OPENING_HOURS: tuple[int, ...] = (8, 9, 10, 11, 14, 15, 16, 17, 18)

# Horizon de prise de rendez-vous, en jours a partir de l'horloge figee.
BOOKING_HORIZON_DAYS = 90


class Base(DeclarativeBase):
    """Base declarative du backend simule."""


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(120))


class Appointment(Base):
    __tablename__ = "appointments"

    appointment_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.patient_id"), index=True)
    practitioner_id: Mapped[str] = mapped_column(String(32), index=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    hour: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime)


def database_url() -> str:
    """URL de la base. SQLite par defaut, PostgreSQL possible pour le profil banc."""
    return os.environ.get("IVR_BENCH_DATABASE_URL", "sqlite+pysqlite:///:memory:")


def is_available(practitioner_id: str, day: date, hour: int) -> bool:
    """Disponibilite theorique d'un praticien, derivee de facon deterministe.

    Le week-end est ferme. Le reste depend d'une empreinte stable : le meme
    praticien a toujours les memes creneaux ouverts le meme jour.
    """
    if day.weekday() >= 5 or hour not in OPENING_HOURS:
        return False
    digest = hashlib.sha256(f"{practitioner_id}|{day.isoformat()}|{hour}".encode()).digest()
    # Environ deux tiers des creneaux ouvrables sont proposes.
    return digest[0] % 3 != 0


class BackendStore:
    """Etat metier simule, remis a zero entre les suites."""

    def __init__(self, practitioners: list[Practitioner], url: str | None = None) -> None:
        self._practitioners = {item.practitioner_id: item for item in practitioners}
        resolved = url or database_url()

        # SQLite en memoire ouvre une base vide par connexion. Sans pool
        # statique, l'API servie sur un autre fil ne verrait aucune table.
        options: dict[str, Any] = {"future": True}
        if ":memory:" in resolved:
            options["poolclass"] = StaticPool
            options["connect_args"] = {"check_same_thread": False}
        self._engine = create_engine(resolved, **options)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)
        Base.metadata.create_all(self._engine)

    @property
    def practitioners(self) -> dict[str, Practitioner]:
        return self._practitioners

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._sessions() as session:
            yield session

    def reset(self, seed: int = 42, patient_count: int = 200) -> None:
        """Reinitialise l'etat : un test ne doit pas influencer le suivant (§6)."""
        with self._sessions() as session:
            session.execute(delete(Appointment))
            session.execute(delete(Patient))
            session.commit()
        self.seed(seed=seed, patient_count=patient_count)

    def seed(self, seed: int = 42, patient_count: int = 200) -> None:
        """Peuple patients et rendez-vous existants de facon deterministe."""
        today = reference_now().date()
        practitioner_ids = sorted(self._practitioners)
        if not practitioner_ids:
            raise ValueError("aucun praticien : le backend ne peut pas etre initialise")

        created_at = reference_now().replace(tzinfo=None)
        with self._sessions() as session:
            for index in range(patient_count):
                patient_id = f"patient_{index + 1:05d}"
                session.add(Patient(patient_id=patient_id, display_name=f"Patient {index + 1}"))

                # Zero a deux rendez-vous existants, repartis autour de la date
                # de reference : certains patients n'ont rien, et c'est un cas de
                # test a part entiere.
                for occurrence in range((index + seed) % 3):
                    offset = 3 + (index * 7 + occurrence * 11 + seed) % 40
                    day = today + timedelta(days=offset)
                    if day.weekday() >= 5:
                        day += timedelta(days=7 - day.weekday())
                    hour = OPENING_HOURS[(index + occurrence) % len(OPENING_HOURS)]
                    practitioner_id = practitioner_ids[
                        (index * 13 + occurrence) % len(practitioner_ids)
                    ]
                    session.add(
                        Appointment(
                            appointment_id=f"appointment_{index + 1:05d}_{occurrence}",
                            patient_id=patient_id,
                            practitioner_id=practitioner_id,
                            day=day,
                            hour=hour,
                            created_at=created_at,
                        )
                    )
            session.commit()

    def horizon(self) -> tuple[date, date]:
        today = reference_now().date()
        return today, today + timedelta(days=BOOKING_HORIZON_DAYS)
