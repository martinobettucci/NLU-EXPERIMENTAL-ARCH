"""Tests de la cle phonetique francaise et de la normalisation des noms."""

from __future__ import annotations

import pytest

from ivr_bench.resolver.normalization import comparable, normalize, strip_titles
from ivr_bench.resolver.phonetics import phonetic_key, phonetic_keys

# Formes que l'oreille ne distingue pas : elles doivent partager une cle.
HOMOPHONES = [
    ("Rey", "Ray"),
    ("Rey", "Reï"),
    ("Bensaïd", "Ben Saïd"),
    ("Ben Saïd", "Ben Said"),
    ("Renault", "Renaud"),
    ("Renaud", "Renaut"),
    ("Dupont", "Dupond"),
    ("Petit", "Petitt"),
    ("Bernard", "Bernart"),
    ("Legrand", "Le Grand"),
    ("Marchand", "Marchant"),
    ("Vidal", "Vidale"),
    ("Leroy", "Leroi"),
    ("Chauvin", "Chovin"),
]

# Noms differents a l'oreille : les confondre couterait une erreur d'identite.
DISTINCT = [
    ("Rey", "Roy"),
    ("Bensaïd", "Bernard"),
    ("Nguyen", "Nogent"),
    ("Le Goff", "Le Guen"),
    ("Martin", "Mercier"),
    ("Dubois", "Dumont"),
]


@pytest.mark.parametrize(("left", "right"), HOMOPHONES)
def test_homophones_share_a_key(left: str, right: str) -> None:
    assert phonetic_key(left) == phonetic_key(right), f"{left} et {right} devraient sonner pareil"


@pytest.mark.parametrize(("left", "right"), DISTINCT)
def test_distinct_names_keep_distinct_keys(left: str, right: str) -> None:
    assert phonetic_key(left) != phonetic_key(right)


def test_key_keeps_enough_substance() -> None:
    """Une cle reduite a une lettre ne discrimine plus rien."""
    for name in ("Rey", "Roy", "Fol", "Bec", "Hue", "Cot"):
        assert len(phonetic_key(name)) >= 2, f"cle trop courte pour {name}"


def test_key_ignores_case_accents_and_separators() -> None:
    assert phonetic_key("BEN-SAÏD") == phonetic_key("ben said")
    assert phonetic_key("D'Angelo") == phonetic_key("Dangelo")


def test_empty_name_has_no_key() -> None:
    assert phonetic_key("") == ""
    assert phonetic_keys(["", "  "]) == ()


def test_keys_are_deduplicated_and_ordered() -> None:
    assert phonetic_keys(["Rey", "Ray", "Roy"]) == (phonetic_key("Rey"), phonetic_key("Roy"))


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("docteur Bensaïd", "bensaid"),
        ("Docteure Rey", "rey"),
        ("madame le docteur Nguyen", "nguyen"),
        ("Dr. Le Goff", "le goff"),
        ("professeur D'Angelo", "d angelo"),
        ("le docteur Martin", "martin"),
        ("Martin", "martin"),
    ],
)
def test_titles_are_stripped_only_for_comparison(spoken: str, expected: str) -> None:
    assert comparable(spoken) == expected


def test_a_title_alone_leaves_nothing_to_resolve() -> None:
    assert strip_titles("docteur") == ""


def test_normalization_keeps_tokens_separate() -> None:
    """L'apostrophe devient un espace : `D'Angelo` reste deux jetons."""
    assert normalize("D'Angelo") == "d angelo"
    assert normalize("Jean-Pierre") == "jean pierre"
