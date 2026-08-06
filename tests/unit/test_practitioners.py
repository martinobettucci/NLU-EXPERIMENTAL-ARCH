"""Tests du catalogue de praticiens et du resolveur (§7, §8)."""

from __future__ import annotations

from collections import Counter

import pytest

from ivr_bench.domain.models import Practitioner
from ivr_bench.generators.practitioners import (
    PROFILE_SIZES,
    generate_practitioners,
    load_practitioners,
)
from ivr_bench.resolver import PractitionerResolver


@pytest.fixture(scope="module")
def practitioners() -> list[Practitioner]:
    return load_practitioners()


@pytest.fixture(scope="module")
def resolver(practitioners: list[Practitioner]) -> PractitionerResolver:
    return PractitionerResolver(practitioners)


# -- catalogue --------------------------------------------------------------


def test_catalog_meets_the_required_size(practitioners: list[Practitioner]) -> None:
    assert len(practitioners) >= 1000


def test_generation_is_deterministic() -> None:
    first = generate_practitioners(seed=7, count=150)
    second = generate_practitioners(seed=7, count=150)
    assert first == second


def test_a_different_seed_changes_the_catalog() -> None:
    assert generate_practitioners(seed=1, count=150) != generate_practitioners(seed=2, count=150)


def test_identifiers_are_opaque_and_unique(practitioners: list[Practitioner]) -> None:
    identifiers = [item.practitioner_id for item in practitioners]
    assert len(set(identifiers)) == len(identifiers)
    assert all(item.startswith("practitioner_") for item in identifiers)


def test_test_split_practitioners_exist(practitioners: list[Practitioner]) -> None:
    """Le test contient des praticiens absents de l'entrainement (§7)."""
    splits = Counter(item.split for item in practitioners)
    assert splits["test"] > 0
    assert splits["train"] > splits["test"]

    train_names = {item.last_name for item in practitioners if item.split == "train"}
    test_only = {item.last_name for item in practitioners if item.split == "test"} - train_names
    assert test_only, "aucun nom n'est propre a la partition de test"


def test_every_difficulty_category_is_represented(practitioners: list[Practitioner]) -> None:
    categories = {item.name_category for item in practitioners}
    for expected in ("homophone", "french_common", "international", "accented", "apostrophe"):
        assert expected in categories, f"categorie absente du catalogue : {expected}"


def test_homophone_groups_are_complete(practitioners: list[Practitioner]) -> None:
    """Toutes les variantes d'un groupe coexistent, sinon l'ambiguite est fictive."""
    groups: dict[str, set[str]] = {}
    for item in practitioners:
        if item.homophone_group:
            groups.setdefault(item.homophone_group, set()).add(item.last_name)
    assert groups
    for group, names in groups.items():
        assert len(names) >= 2, f"le groupe {group} n'a qu'une variante"


def test_catalog_mixes_unique_and_shared_surnames(practitioners: list[Practitioner]) -> None:
    """Les deux cas doivent exister : sinon une seule reponse serait possible."""
    counts = Counter(item.last_name for item in practitioners)
    assert any(value == 1 for value in counts.values())
    assert any(value > 1 for value in counts.values())


def test_profiles_are_ordered() -> None:
    assert PROFILE_SIZES["dev"] < PROFILE_SIZES["smoke"] < PROFILE_SIZES["full"]
    assert PROFILE_SIZES["full"] >= 1000


# -- resolveur --------------------------------------------------------------


def test_absent_name_is_missing_not_invented(resolver: PractitionerResolver) -> None:
    assert resolver.resolve(None).status == "missing"
    assert resolver.resolve("   ").status == "missing"
    assert resolver.resolve("docteur").status == "missing"


def test_unknown_name_is_not_found(resolver: PractitionerResolver) -> None:
    outcome = resolver.resolve("docteur Zzyzxwq")
    assert outcome.status == "not_found"
    assert outcome.practitioner_id is None


def test_unique_surname_resolves(
    resolver: PractitionerResolver, practitioners: list[Practitioner]
) -> None:
    counts = Counter(item.last_name for item in practitioners)
    unique = next(
        item
        for item in practitioners
        if counts[item.last_name] == 1 and item.homophone_group is None
    )
    outcome = resolver.resolve(f"docteur {unique.last_name}")
    assert outcome.status == "resolved"
    assert outcome.practitioner_id == unique.practitioner_id


def test_shared_surname_asks_for_clarification(
    resolver: PractitionerResolver, practitioners: list[Practitioner]
) -> None:
    counts = Counter(item.last_name for item in practitioners)
    shared = next(item for item in practitioners if counts[item.last_name] > 1)
    outcome = resolver.resolve(f"docteur {shared.last_name}")
    assert outcome.status == "ambiguous"
    assert outcome.practitioner_id is None
    assert len(outcome.candidates) >= 2


def test_homophones_are_never_resolved_silently(
    resolver: PractitionerResolver, practitioners: list[Practitioner]
) -> None:
    """Sur un canal vocal, deux orthographes homophones ne se departagent pas."""
    grouped = [item for item in practitioners if item.homophone_group]
    assert grouped
    for item in grouped[:12]:
        outcome = resolver.resolve(f"docteur {item.last_name}")
        assert outcome.status == "ambiguous", (
            f"{item.last_name} a ete resolu unilateralement malgre ses homophones"
        )


def test_specialty_disambiguates_a_shared_surname(
    resolver: PractitionerResolver, practitioners: list[Practitioner]
) -> None:
    counts = Counter(item.last_name for item in practitioners)
    shared = next(item for item in practitioners if counts[item.last_name] > 1)
    outcome = resolver.resolve(f"docteur {shared.last_name}", specialty=shared.specialty)
    assert outcome.status == "resolved"
    assert outcome.practitioner_id == shared.practitioner_id


def test_titles_do_not_change_the_outcome(
    resolver: PractitionerResolver, practitioners: list[Practitioner]
) -> None:
    counts = Counter(item.last_name for item in practitioners)
    unique = next(
        item
        for item in practitioners
        if counts[item.last_name] == 1 and item.homophone_group is None
    )
    forms = [
        unique.last_name,
        f"docteur {unique.last_name}",
        f"Dr {unique.last_name}",
        f"madame le docteur {unique.last_name}",
        f"le docteur {unique.first_name} {unique.last_name}",
    ]
    for form in forms:
        assert resolver.resolve(form).practitioner_id == unique.practitioner_id, form


def test_an_impossible_filter_is_ignored_rather_than_failing(
    resolver: PractitionerResolver, practitioners: list[Practitioner]
) -> None:
    """Une specialite mal comprise ne doit pas transformer un nom connu en inconnu."""
    counts = Counter(item.last_name for item in practitioners)
    unique = next(
        item
        for item in practitioners
        if counts[item.last_name] == 1 and item.homophone_group is None
    )
    outcome = resolver.resolve(f"docteur {unique.last_name}", specialty="astrophysique")
    assert outcome.status == "resolved"
    assert outcome.practitioner_id == unique.practitioner_id


def test_empty_catalog_is_rejected() -> None:
    with pytest.raises(ValueError, match="catalogue non vide"):
        PractitionerResolver([])
