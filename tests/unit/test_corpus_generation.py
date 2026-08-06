"""Tests des generateurs de corpus et du controle de contamination (§10)."""

from __future__ import annotations

import pytest

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.validation import OutputValidator
from ivr_bench.generators.corpus import (
    PROFILE_VOLUMES,
    build_corpus,
    deduplicate,
    load_split,
    normalized,
)
from ivr_bench.generators.utterances import GeneratedCase, load_templates, semantic_core


@pytest.fixture(scope="module")
def corpus() -> dict[str, list[GeneratedCase]]:
    built = build_corpus(seed=7, profile="dev")
    cleaned, _ = deduplicate(built)
    return cleaned


# -- independance des deux generateurs --------------------------------------


def test_generators_share_no_template_family() -> None:
    """L'anti-fuite repose sur la construction, pas sur un filtrage (§10.3)."""
    bank_a = load_templates("a")
    bank_b = load_templates("b")
    families_a = {family.id for family in bank_a.families}
    families_b = {family.id for family in bank_b.families}
    assert families_a.isdisjoint(families_b)


def test_generators_share_no_pattern() -> None:
    bank_a = load_templates("a")
    bank_b = load_templates("b")
    patterns_a = {pattern for family in bank_a.families for pattern in family.patterns}
    patterns_b = {pattern for family in bank_b.families for pattern in family.patterns}
    assert patterns_a.isdisjoint(patterns_b)


def test_generators_share_no_lexicon_entry() -> None:
    bank_a = load_templates("a")
    bank_b = load_templates("b")
    for key in ("openers", "dates", "times"):
        assert set(bank_a.lexicon.get(key, ())).isdisjoint(set(bank_b.lexicon.get(key, ()))), key


def test_every_function_is_covered_by_both_generators() -> None:
    names = set(default_catalog().names)
    for generator in ("a", "b"):
        covered = {family.function for family in load_templates(generator).families}
        assert covered == names, f"generateur {generator} : fonctions manquantes"


# -- proprietes du corpus ---------------------------------------------------


def test_no_test_utterance_appears_in_training(corpus: dict[str, list[GeneratedCase]]) -> None:
    test_keys = {normalized(case.utterance) for case in corpus["test"]}
    for split in ("index", "train", "validation", "contrastive"):
        overlap = {normalized(case.utterance) for case in corpus[split]} & test_keys
        assert not overlap, f"{split} partage {len(overlap)} enonces avec le test"


def test_test_split_only_cites_unseen_practitioners(
    corpus: dict[str, list[GeneratedCase]],
) -> None:
    """Un routeur ne peut pas reciter une liste de noms apprise (§7)."""
    from ivr_bench.generators.practitioners import load_practitioners

    train_names = {item.last_name for item in load_practitioners() if item.split == "train"}
    for case in corpus["test"]:
        spoken = case.expected.arguments.get("practitioner_name")
        if spoken:
            assert not any(name in spoken for name in train_names), spoken


def test_expected_calls_all_pass_output_validation(
    corpus: dict[str, list[GeneratedCase]],
) -> None:
    """Une cible invalide rendrait la comparaison entre architectures faussee."""
    validator = OutputValidator(default_catalog())
    for split, cases in corpus.items():
        for case in cases:
            outcome = validator.validate(
                {"name": case.expected.tool_name, "arguments": case.expected.arguments}
            )
            assert outcome.validity != "invalid", f"{split}/{case.id} : {outcome.errors}"


def test_unspoken_arguments_are_null_not_absent(
    corpus: dict[str, list[GeneratedCase]],
) -> None:
    """Distinguer 'non exprime' de 'hallucine' exige que la cle existe (§12)."""
    catalog = default_catalog()
    for case in corpus["test"]:
        definition = catalog.get(case.expected.tool_name)
        if not definition.executable:
            continue
        assert set(case.expected.arguments) == set(definition.parameter_names)


def test_generation_is_deterministic() -> None:
    first, _ = deduplicate(build_corpus(seed=11, profile="dev"))
    second, _ = deduplicate(build_corpus(seed=11, profile="dev"))
    assert [case.utterance for case in first["test"]] == [case.utterance for case in second["test"]]


def test_a_different_seed_changes_the_corpus() -> None:
    first, _ = deduplicate(build_corpus(seed=11, profile="dev"))
    second, _ = deduplicate(build_corpus(seed=12, profile="dev"))
    assert [case.utterance for case in first["test"]] != [case.utterance for case in second["test"]]


def test_contrastive_cases_declare_the_function_they_challenge(
    corpus: dict[str, list[GeneratedCase]],
) -> None:
    assert corpus["contrastive"]
    for case in corpus["contrastive"]:
        target = case.metadata.get("contrast_for")
        assert target, f"{case.id} ne declare pas la fonction qu'il piege"
        # Un contrastif porte l'etiquette de sa VRAIE fonction, celle du voisin.
        assert case.expected.tool_name != target


def test_deduplication_reports_what_it_removed() -> None:
    _, report = deduplicate(build_corpus(seed=7, profile="dev"))
    assert set(report) >= {"within_split", "leaked_into_training", "test_size", "total_removed"}
    assert report["total_removed"] == sum(report["within_split"].values()) + sum(
        report["leaked_into_training"].values()
    )


def test_semantic_core_ignores_politeness() -> None:
    """Deux variantes qui ne different que par la politesse ont le meme noyau."""
    assert semantic_core("où est le parking ?") == semantic_core(
        "où est le parking s'il vous plait ?"
    )
    assert semantic_core("où est le parking ?") != semantic_core("quels sont vos horaires ?")


def test_profiles_are_ordered() -> None:
    assert PROFILE_VOLUMES["dev"]["test"] < PROFILE_VOLUMES["smoke"]["test"]
    assert PROFILE_VOLUMES["smoke"]["test"] < PROFILE_VOLUMES["full"]["test"]
    assert PROFILE_VOLUMES["full"]["index"] >= 300


# -- corpus versionne -------------------------------------------------------


@pytest.mark.parametrize("split", ["index", "train", "validation", "contrastive", "test"])
def test_versioned_split_is_readable(split: str) -> None:
    cases = load_split(split)
    assert cases, f"partition vide : {split}"
    assert all(case.split == split for case in cases)
