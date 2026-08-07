"""Extraction d'arguments par regles deterministes.

Sert la baseline A0 et le retriever seul A5 (§9). Une regle n'invente jamais :
un argument non exprime reste `None`, et c'est le gestionnaire de dialogue qui
posera la question. Un extracteur qui comblerait les vides ferait mieux paraitre
la baseline tout en produisant exactement l'erreur que les metriques punissent
le plus fort (§12).
"""

from __future__ import annotations

import re

from ivr_bench.domain.dates import MONTHS, TIME_WINDOWS, WEEKDAYS
from ivr_bench.resolver.phonetics import strip_accents

# Titres introduisant un nom de praticien. Le titre fait partie de l'argument
# attendu : le corpus le conserve, le resolveur le retirera plus tard.
_TITLE = (
    r"(?:madame\s+le\s+docteur|monsieur\s+le\s+docteur|le\s+docteur"
    r"|la\s+docteure|docteure|docteur|professeure|professeur|dr\.?|pr\.?)"
)

# Un nom peut etre compose, accentue, apostrophe : « Jean-Pierre Le Goff »,
# « D'Angelo », « Ben Said ».
_NAME = r"[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]*)*"

_PRACTITIONER = re.compile(rf"\b({_TITLE}\s+{_NAME})", re.IGNORECASE)

_RELATIVE_DATES = (
    "apres-demain",
    "après-demain",
    "aujourd'hui",
    "demain",
    "la semaine prochaine",
    "le mois prochain",
    "fin de semaine",
    "lundi prochain",
)

_EXPLICIT_TIME = re.compile(r"\b\d{1,2}\s*h(?:\s*\d{2})?\b", re.IGNORECASE)
_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b")
_DAY_MONTH = re.compile(rf"\b(?:le\s+)?\d{{1,2}}(?:er)?\s+({'|'.join(MONTHS)})\b", re.IGNORECASE)
_WITHIN = re.compile(r"\bdans\s+(?:\w+)\s+(?:jours?|semaines?|mois)\b", re.IGNORECASE)


def extract_practitioner(utterance: str) -> str | None:
    """Nom prononce, titre compris, ou None s'il n'y en a pas."""
    match = _PRACTITIONER.search(utterance)
    if not match:
        return None
    return " ".join(match.group(1).split()).rstrip(" ,.;:!?")


def extract_date(utterance: str) -> str | None:
    """Expression de date telle que prononcee."""
    lowered = strip_accents(utterance).lower()

    for phrase in _RELATIVE_DATES:
        plain = strip_accents(phrase).lower()
        if plain in lowered:
            return phrase

    for pattern in (_DAY_MONTH, _NUMERIC_DATE, _WITHIN):
        match = pattern.search(utterance)
        if match:
            return match.group(0).strip()

    for day in WEEKDAYS:
        if re.search(rf"\b{day}\b", lowered):
            return day
    return None


def extract_time(utterance: str) -> str | None:
    """Expression horaire telle que prononcee."""
    match = _EXPLICIT_TIME.search(utterance)
    if match:
        return match.group(0).strip()

    lowered = strip_accents(utterance).lower()
    # Du libelle le plus long au plus court : « midi » ne doit pas l'emporter a
    # l'interieur de « apres midi ».
    for label in sorted(TIME_WINDOWS, key=len, reverse=True):
        if label in lowered.replace("-", " "):
            return label
    return None


def extract_specialty(utterance: str, specialties: list[str]) -> str | None:
    """Specialite citee, parmi celles que le centre propose reellement."""
    lowered = strip_accents(utterance).lower()
    # Les libelles longs d'abord : « medecine generale » avant « medecine ».
    for specialty in sorted(specialties, key=len, reverse=True):
        if strip_accents(specialty).lower() in lowered:
            return specialty
    return None
