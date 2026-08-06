"""Interpretation des dates et heures prononcees en francais.

Le routeur ne normalise pas les dates : il transmet ce qui a ete dit, tel quel
(« mardi », « la semaine prochaine », « demain matin »). La conversion en date
reelle appartient au moteur deterministe, et elle est mesuree separement par les
metriques `date normalization accuracy` et `time normalization accuracy` (§17.2).

Tout est calcule par rapport a l'horloge figee : sans cela, la meme campagne
rejouee un autre jour changerait de reponses attendues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from ivr_bench.domain.clock import reference_now
from ivr_bench.resolver.phonetics import strip_accents

WEEKDAYS: dict[str, int] = {
    "lundi": 0,
    "mardi": 1,
    "mercredi": 2,
    "jeudi": 3,
    "vendredi": 4,
    "samedi": 5,
    "dimanche": 6,
}

MONTHS: dict[str, int] = {
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}

# Plages horaires de la journee, en heures pleines.
TIME_WINDOWS: dict[str, tuple[int, int]] = {
    "matin": (8, 12),
    "midi": (12, 14),
    "apres midi": (14, 18),
    "soir": (18, 20),
    "journee": (8, 20),
}

_NUMBER_WORDS: dict[str, int] = {
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
}

_EXPLICIT_TIME = re.compile(r"\b(\d{1,2})\s*(?:h|heures?)\s*(\d{2})?\b")
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?\b")
_DAY_MONTH = re.compile(r"\b(\d{1,2})(?:er)?\s+([a-z]+)\b")


@dataclass(frozen=True)
class SpokenDateTime:
    """Interpretation d'une date et d'un moment prononces."""

    day: date | None = None
    start_hour: int | None = None
    end_hour: int | None = None
    #: Vrai quand l'appelant a exprime une periode plutot qu'un jour precis.
    is_range: bool = False
    range_end: date | None = None

    @property
    def is_empty(self) -> bool:
        return self.day is None and self.start_hour is None


def _normalize(text: str) -> str:
    lowered = strip_accents(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _next_weekday(reference: date, weekday: int) -> date:
    """Prochaine occurrence d'un jour de la semaine.

    « mardi » prononce un mardi designe le mardi suivant, pas le jour meme : un
    appelant qui demande un rendez-vous « mardi » ne parle pas de l'instant
    present.
    """
    delta = (weekday - reference.weekday()) % 7
    if delta == 0:
        delta = 7
    return reference + timedelta(days=delta)


def parse_time(text: str) -> tuple[int | None, int | None]:
    """Extrait une plage horaire d'un enonce."""
    normalized = _normalize(text)

    explicit = _EXPLICIT_TIME.search(normalized)
    if explicit:
        hour = int(explicit.group(1))
        if 0 <= hour <= 23:
            return hour, hour + 1

    # Du libelle le plus long au plus court : « midi » ne doit pas l'emporter
    # a l'interieur de « apres midi ».
    for label in sorted(TIME_WINDOWS, key=len, reverse=True):
        if label in normalized:
            return TIME_WINDOWS[label]
    return None, None


def parse_date(text: str, reference: date | None = None) -> tuple[date | None, date | None]:
    """Extrait une date, et une date de fin lorsque l'enonce designe une periode."""
    today = reference or reference_now().date()
    # Les separateurs de '12/03' doivent survivre a la normalisation, qui les
    # remplacerait par des espaces.
    raw = strip_accents(text).lower()
    normalized = _normalize(text)
    if not normalized:
        return None, None

    if "avant hier" in normalized:
        return today - timedelta(days=2), None
    if "apres demain" in normalized:
        return today + timedelta(days=2), None
    if "demain" in normalized:
        return today + timedelta(days=1), None
    if "hier" in normalized:
        return today - timedelta(days=1), None
    if "aujourd hui" in normalized or "ce jour" in normalized:
        return today, None

    # Periodes : elles produisent un intervalle, pas un jour unique.
    if "semaine prochaine" in normalized:
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=6)
    if "cette semaine" in normalized:
        start = today - timedelta(days=today.weekday())
        return start, start + timedelta(days=6)
    if "mois prochain" in normalized:
        first = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        return first, (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)

    within = re.search(r"\bdans\s+([a-z]+|\d+)\s+(jours?|semaines?|mois)\b", normalized)
    if within:
        token = within.group(1)
        amount = int(token) if token.isdigit() else _NUMBER_WORDS.get(token, 1)
        unit = within.group(2)
        if unit.startswith("jour"):
            return today + timedelta(days=amount), None
        if unit.startswith("semaine"):
            return today + timedelta(weeks=amount), None
        return today + timedelta(days=30 * amount), None

    numeric = _NUMERIC_DATE.search(raw)
    if numeric:
        day, month = int(numeric.group(1)), int(numeric.group(2))
        year = int(numeric.group(3) or today.year)
        if year < 100:
            year += 2000
        parsed = _safe_date(year, month, day)
        if parsed:
            return parsed, None

    day_month = _DAY_MONTH.search(normalized)
    if day_month and day_month.group(2) in MONTHS:
        day = int(day_month.group(1))
        month = MONTHS[day_month.group(2)]
        year = today.year if month >= today.month else today.year + 1
        parsed = _safe_date(year, month, day)
        if parsed:
            return parsed, None

    for label, weekday in WEEKDAYS.items():
        if label in normalized:
            return _next_weekday(today, weekday), None

    return None, None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse(text: str | None, reference: date | None = None) -> SpokenDateTime:
    """Interprete conjointement la date et le moment d'un enonce."""
    if not text:
        return SpokenDateTime()

    parsed_date, range_end = parse_date(text, reference=reference)
    start_hour, end_hour = parse_time(text)
    return SpokenDateTime(
        day=parsed_date,
        start_hour=start_hour,
        end_hour=end_hour,
        is_range=range_end is not None,
        range_end=range_end,
    )
