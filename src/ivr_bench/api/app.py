"""API de demonstration du serveur vocal simule (§34, profil prod).

Cette surface sert la demonstration et les tests d'integration. Elle ne publie
aucun resultat de campagne et ne contient aucune donnee reelle. Le `patient_id`
provient de l'en-tete de session authentifiee, jamais du corps de la requete ni
d'une transcription.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from ivr_bench.backend.service import BackendService
from ivr_bench.backend.store import BackendStore
from ivr_bench.domain.catalog import default_catalog, load_general_information
from ivr_bench.domain.clock import reference_now
from ivr_bench.domain.dates import parse
from ivr_bench.generators.practitioners import load_practitioners
from ivr_bench.resolver import PractitionerResolver

app = FastAPI(
    title="Serveur vocal simule",
    description="Demonstration experimentale. Donnees entierement synthetiques.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def _service() -> tuple[BackendService, PractitionerResolver]:
    practitioners = load_practitioners()
    store = BackendStore(practitioners)
    store.seed()
    return BackendService(store), PractitionerResolver(practitioners)


class CreationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    practitioner_name: str
    preferred_date: str | None = None
    preferred_time: str | None = None
    specialty: str | None = None
    #: La creation n'a lieu qu'apres confirmation explicite de l'appelant.
    confirmed: bool = False
    dry_run: bool = False


@app.get("/health")
def health() -> dict[str, Any]:
    catalog = default_catalog()
    return {
        "status": "ok",
        "catalog_version": catalog.version,
        "functions": len(catalog.functions),
        "reference_time": reference_now().isoformat(),
    }


@app.get("/information/{topic}")
def information(topic: str) -> dict[str, str]:
    topics = load_general_information()["topics"]
    if topic not in topics:
        raise HTTPException(status_code=404, detail="sujet inconnu")
    return {"topic": topic, "answer": topics[topic]}


@app.get("/practitioners/resolve")
def resolve(name: str, specialty: str | None = None) -> dict[str, Any]:
    _, resolver = _service()
    outcome = resolver.resolve(name, specialty=specialty)
    return outcome.model_dump()


@app.get("/appointments")
def appointments(
    x_patient_id: Annotated[str | None, Header()] = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> dict[str, Any]:
    service, _ = _service()
    result = service.list_appointments(x_patient_id, date_from=date_from, date_to=date_to)
    if result.status == "authentication_required":
        raise HTTPException(status_code=401, detail="session patient requise")
    return {
        "status": result.status,
        "appointments": [
            {
                "appointment_id": item.appointment_id,
                "practitioner_name": item.practitioner_name,
                "day": item.day.isoformat(),
                "hour": item.hour,
            }
            for item in result.appointments
        ],
    }


@app.post("/appointments")
def create(
    payload: CreationRequest,
    x_patient_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    service, resolver = _service()

    resolution = resolver.resolve(payload.practitioner_name, specialty=payload.specialty)
    # Une identite incertaine ne declenche pas d'ecriture : on renvoie les
    # candidats pour que l'appelant tranche.
    if resolution.status != "resolved" or resolution.practitioner_id is None:
        return {"status": resolution.status, "resolution": resolution.model_dump()}

    when = parse(f"{payload.preferred_date or ''} {payload.preferred_time or ''}".strip())
    availability = service.availability(resolution.practitioner_id, when=when)
    if not availability.succeeded:
        return {"status": availability.status}

    _, day, hour = availability.proposed[0]
    result = service.create_appointment(
        x_patient_id,
        resolution.practitioner_id,
        day,
        hour,
        confirmed=payload.confirmed,
        dry_run=payload.dry_run,
    )
    if result.status == "authentication_required":
        raise HTTPException(status_code=401, detail="session patient requise")
    return {
        "status": result.status,
        "practitioner_id": resolution.practitioner_id,
        "day": day.isoformat(),
        "hour": hour,
        "detail": result.detail,
    }
