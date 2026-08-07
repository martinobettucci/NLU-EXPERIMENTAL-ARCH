"""Composition classifieur + DIET : ce qui doit tenir sans aucun poids.

Le realignement des roles decide de ce que la composition conserve des entites
de DIET. Il se teste sans charger ni l'encodeur ni le modele Rasa : c'est une
fonction du catalogue et d'une liste d'entites.
"""

from __future__ import annotations

from typing import Any

import pytest

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import ToolDefinition
from ivr_bench.routers import available
from ivr_bench.routers.classifier.enumerations import EnumerationLearner
from ivr_bench.routers.composite.arbitration import DIET, RULES, ArbitratedRouter
from ivr_bench.routers.composite.router import ClassifierDietRouter


class _Extractor(ClassifierDietRouter):
    """Le realignement seul, sans classifieur ni sidecar."""

    def __init__(self) -> None:
        # On saute volontairement l'initialisation du parent : ni encodeur ni
        # modele DIET ne sont necessaires pour eprouver la fusion d'arguments.
        self._catalog = default_catalog()
        self._specialties = []


class _Arbiter(ArbitratedRouter):
    """L'arbitrage seul : les deux sources sont fournies, pas calculees."""

    def __init__(self, preference: dict[str, str], diet: dict[str, Any]) -> None:
        self._catalog = default_catalog()
        self._specialties = []
        self._preference = preference
        self._enumerations = EnumerationLearner()
        self._diet_values = diet

    def _diet_arguments(
        self, definition: ToolDefinition, utterance: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        values = dict.fromkeys(definition.parameter_names)
        values.update(self._diet_values)
        return values


@pytest.fixture(scope="module")
def extractor() -> _Extractor:
    return _Extractor()


def test_both_compositions_are_registered() -> None:
    assert {
        "classifier_diet",
        "classifier_diet_rules",
        "classifier_diet_arbitrated",
        "classifier_diet_enum",
    } <= set(available())


def test_the_preferred_source_wins_argument_by_argument() -> None:
    definition = default_catalog().get("request_new_appointment")
    arbiter = _Arbiter(
        preference={"practitioner_name": DIET, "preferred_date": RULES},
        diet={"practitioner_name": "Bensaïd", "preferred_date": "2026-01-01"},
    )
    metadata: dict[str, Any] = {}
    values = arbiter._compose(
        definition, "rendez-vous avec le docteur Bensaïd mardi", True, metadata
    )
    # Le nom vient de DIET, la date des regles : c'est ce que la validation a
    # mesure, argument par argument.
    assert values["practitioner_name"] == "Bensaïd"
    assert values["preferred_date"] != "2026-01-01"
    assert metadata["sources"]["practitioner_name"] == DIET
    assert metadata["sources"]["preferred_date"] == RULES


def test_the_other_source_is_consulted_when_the_preferred_one_is_empty() -> None:
    definition = default_catalog().get("request_new_appointment")
    arbiter = _Arbiter(preference={"practitioner_name": DIET}, diet={})
    metadata: dict[str, Any] = {}
    values = arbiter._compose(definition, "rendez-vous avec le docteur Rey", True, metadata)
    assert values["practitioner_name"] is not None
    assert metadata["sources"]["practitioner_name"] == RULES


def test_exact_entity_names_are_kept(extractor: _Extractor) -> None:
    definition = default_catalog().get("request_new_appointment")
    arguments, realigned = extractor._from_entities(
        definition,
        [
            {"entity": "practitioner_name", "value": "Rey"},
            {"entity": "preferred_date", "value": "mardi"},
        ],
    )
    assert arguments["practitioner_name"] == "Rey"
    assert arguments["preferred_date"] == "mardi"
    assert realigned == 0


def test_a_date_labelled_for_another_function_is_realigned(extractor: _Extractor) -> None:
    """DIET a vu la date, mais sous le nom de l'intention qu'il avait choisie."""
    definition = default_catalog().get("request_new_appointment")
    arguments, realigned = extractor._from_entities(
        definition, [{"entity": "date_from", "value": "mardi"}]
    )
    assert arguments["preferred_date"] == "mardi"
    assert realigned == 1


def test_a_single_value_per_role(extractor: _Extractor) -> None:
    """Deux dates ne remplissent pas deux champs : la seconde est ecartee."""
    definition = default_catalog().get("list_appointments")
    arguments, _ = extractor._from_entities(
        definition,
        [
            {"entity": "current_date", "value": "mardi"},
            {"entity": "preferred_new_date", "value": "vendredi"},
        ],
    )
    assert arguments["date_from"] == "mardi"
    assert arguments["date_to"] is None


def test_unknown_entities_are_dropped(extractor: _Extractor) -> None:
    definition = default_catalog().get("list_appointments")
    arguments, realigned = extractor._from_entities(
        definition, [{"entity": "appointment_reference", "value": "RDV-12"}]
    )
    assert set(arguments) == set(definition.parameter_names)
    assert all(value is None for value in arguments.values())
    assert realigned == 0


def test_enumerated_arguments_come_from_the_definition(extractor: _Extractor) -> None:
    definition = default_catalog().get("emergency_handoff")
    arguments, _ = extractor._from_entities(definition, [])
    assert arguments["reason_category"] is not None
