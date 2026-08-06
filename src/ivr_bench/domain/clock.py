"""Horloge figee du banc d'essai (§6).

« mardi matin », « la semaine prochaine », « demain » n'ont de sens que par
rapport a un instant. Si cet instant est l'heure courante, la meme campagne
rejouee demain produit d'autres dates attendues et les resultats cessent d'etre
comparables. La reference est donc figee, versionnee et configurable.
"""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "Europe/Paris"

# Lundi 2 mars 2026, 9 h 00, heure de Paris. Un lundi matin en semaine ouvree :
# toutes les dates relatives du corpus tombent alors sur des jours ouvrables.
DEFAULT_REFERENCE = "2026-03-02T09:00:00"


def timezone(name: str | None = None) -> ZoneInfo:
    return ZoneInfo(name or os.environ.get("IVR_BENCH_TIMEZONE", DEFAULT_TIMEZONE))


def reference_now(reference: str | None = None, tz_name: str | None = None) -> datetime:
    """Instant de reference des dates relatives.

    Surchargeable par `IVR_BENCH_REFERENCE_TIME` pour rejouer une campagne dans
    d'autres conditions calendaires, jamais pour suivre l'heure reelle.
    """
    raw = reference or os.environ.get("IVR_BENCH_REFERENCE_TIME", DEFAULT_REFERENCE)
    return datetime.fromisoformat(raw).replace(tzinfo=timezone(tz_name))
