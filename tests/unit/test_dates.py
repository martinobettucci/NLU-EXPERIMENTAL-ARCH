"""Tests de l'interpretation des dates et heures prononcees."""

from __future__ import annotations

from datetime import date

import pytest

from ivr_bench.domain.clock import reference_now
from ivr_bench.domain.dates import parse, parse_date, parse_time

# Lundi 2 mars 2026, reference figee du banc d'essai.
REFERENCE = date(2026, 3, 2)


def test_frozen_clock_is_a_monday() -> None:
    """Les dates relatives du corpus doivent tomber en semaine ouvree."""
    assert reference_now().date().weekday() == 0


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("demain", date(2026, 3, 3)),
        ("apres-demain", date(2026, 3, 4)),
        ("aujourd'hui", REFERENCE),
        ("mardi", date(2026, 3, 3)),
        ("vendredi", date(2026, 3, 6)),
        ("le 12 mars", date(2026, 3, 12)),
        ("le 1er avril", date(2026, 4, 1)),
        ("12/03", date(2026, 3, 12)),
        ("dans deux semaines", date(2026, 3, 16)),
        ("dans 3 jours", date(2026, 3, 5)),
    ],
)
def test_relative_dates_resolve_against_the_frozen_clock(spoken: str, expected: date) -> None:
    parsed, _ = parse_date(spoken, reference=REFERENCE)
    assert parsed == expected


def test_a_weekday_never_means_today() -> None:
    """« lundi » prononce un lundi designe le lundi suivant."""
    parsed, _ = parse_date("lundi", reference=REFERENCE)
    assert parsed == date(2026, 3, 9)


def test_a_period_produces_an_interval() -> None:
    start, end = parse_date("la semaine prochaine", reference=REFERENCE)
    assert start == date(2026, 3, 9)
    assert end == date(2026, 3, 15)


def test_unrecognised_date_stays_none_rather_than_guessed() -> None:
    """Une date non comprise n'est pas inventee : elle reste absente."""
    assert parse_date("quand vous voulez", reference=REFERENCE) == (None, None)


@pytest.mark.parametrize(
    ("spoken", "start", "end"),
    [
        ("le matin", 8, 12),
        ("l'apres-midi", 14, 18),
        ("en soiree", 18, 20),
        ("le soir", 18, 20),
        ("a 14h", 14, 15),
        ("a 9 heures", 9, 10),
        ("14h30", 14, 15),
    ],
)
def test_time_windows(spoken: str, start: int | None, end: int | None) -> None:
    assert parse_time(spoken) == (start, end)


def test_date_and_time_are_parsed_together() -> None:
    parsed = parse("mardi matin", reference=REFERENCE)
    assert parsed.day == date(2026, 3, 3)
    assert (parsed.start_hour, parsed.end_hour) == (8, 12)
    assert parsed.is_range is False


def test_empty_input_is_empty() -> None:
    assert parse(None).is_empty
    assert parse("").is_empty
