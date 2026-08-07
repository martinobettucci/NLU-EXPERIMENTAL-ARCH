"""Tests de contrat communs a toutes les architectures (§32.2).

Chaque routeur, quelle que soit sa technologie, doit reussir exactement ces
tests. Ils ne mesurent pas la qualite — c'est le role du benchmark — mais les
invariants sans lesquels une comparaison n'aurait aucun sens : produire une
sortie valide, respecter le sous-catalogue transmis, ne jamais fabriquer une
identite, ne jamais inventer un argument absent.
"""

from __future__ import annotations

import pytest

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import RouterPrediction, SessionContext
from ivr_bench.domain.validation import OutputValidator
from ivr_bench.routers import available, create
from ivr_bench.routers.base import Router

# Les architectures adossees a un index ou a des poids sont marquees : elles ne
# tournent que lorsque les artefacts reels sont presents, jamais sur substitut.
ROUTERS_WITHOUT_WEIGHTS = ["rules"]
ROUTERS_WITH_WEIGHTS = ["embedding_only", "needle_full", "functiongemma_zero_shot"]

UTTERANCES = [
    "je voudrais un rendez-vous avec le docteur Rey mardi matin",
    "quels sont mes prochains rendez-vous ?",
    "à quelle heure ouvre le cabinet ?",
    "déplacez mon rendez-vous de demain à vendredi",
    "j'ai très mal à la poitrine et je respire mal",
    "quel temps fera-t-il demain ?",
    "je veux parler à quelqu'un",
    "",
    "euh… bonjour, enfin, voilà",
]


@pytest.fixture(scope="module")
def session() -> SessionContext:
    return SessionContext(patient_id="patient_00002")


def _router(name: str) -> Router:
    try:
        return create(name)
    except FileNotFoundError as error:  # index absent
        pytest.skip(str(error))
    except RuntimeError as error:  # poids absents
        pytest.skip(str(error))


def _predictions(name: str, session: SessionContext) -> list[RouterPrediction]:
    router = _router(name)
    tools = list(default_catalog().functions)
    return [router.predict(utterance, session, tools) for utterance in UTTERANCES]


def test_every_registered_router_is_reachable() -> None:
    registered = set(available())
    assert set(ROUTERS_WITHOUT_WEIGHTS) | set(ROUTERS_WITH_WEIGHTS) <= registered


@pytest.mark.parametrize("name", ROUTERS_WITHOUT_WEIGHTS)
def test_contract_without_weights(name: str, session: SessionContext) -> None:
    _assert_contract(name, session)


@pytest.mark.models
@pytest.mark.parametrize("name", ROUTERS_WITH_WEIGHTS)
def test_contract_with_real_weights(name: str, session: SessionContext) -> None:
    _assert_contract(name, session)


def _assert_contract(name: str, session: SessionContext) -> None:
    catalog = default_catalog()
    validator = OutputValidator(catalog)
    predictions = _predictions(name, session)

    for utterance, prediction in zip(UTTERANCES, predictions, strict=True):
        context = f"{name} / {utterance!r}"

        # 1. La sortie appartient au catalogue, ou bien le routeur s'abstient.
        assert prediction.tool_name is None or prediction.tool_name in catalog.names, context

        # 2. Elle franchit la chaine de validation : un routeur qui produirait
        #    une sortie invalide fausserait toutes les metriques d'arguments.
        if prediction.tool_name is not None:
            outcome = validator.validate(
                {"name": prediction.tool_name, "arguments": prediction.arguments}
            )
            assert outcome.validity != "invalid", f"{context} : {outcome.errors}"

        # 3. Aucune identite fabriquee (§25.3, §25.4).
        assert "patient_id" not in prediction.arguments, context
        assert "practitioner_id" not in prediction.arguments, context

        # 4. La latence est mesuree, pas laissee a zero.
        assert prediction.latency_ms >= 0.0, context


@pytest.mark.parametrize("name", ROUTERS_WITHOUT_WEIGHTS)
def test_router_honours_the_subcatalogue(name: str, session: SessionContext) -> None:
    """Le coeur des architectures hybrides : on ne choisit pas hors des candidats."""
    router = _router(name)
    catalog = default_catalog()
    subset = list(catalog.subset(["list_appointments", "no_tool"]))

    for utterance in UTTERANCES:
        prediction = router.predict(utterance, session, subset)
        assert prediction.tool_name in {"list_appointments", "no_tool", None}, utterance


@pytest.mark.parametrize("name", ROUTERS_WITHOUT_WEIGHTS)
def test_router_does_not_invent_unspoken_arguments(name: str, session: SessionContext) -> None:
    """« Je voudrais prendre rendez-vous » ne contient ni date ni praticien."""
    router = _router(name)
    prediction = router.predict(
        "je voudrais prendre rendez-vous", session, list(default_catalog().functions)
    )
    if prediction.tool_name == "request_new_appointment":
        for argument in ("practitioner_name", "preferred_date", "preferred_time"):
            assert prediction.arguments.get(argument) is None, argument


@pytest.mark.parametrize("name", ROUTERS_WITHOUT_WEIGHTS)
def test_router_extracts_what_was_actually_spoken(name: str, session: SessionContext) -> None:
    router = _router(name)
    prediction = router.predict(
        "je voudrais un rendez-vous avec le docteur Rey mardi matin",
        session,
        list(default_catalog().functions),
    )
    if prediction.tool_name == "request_new_appointment":
        spoken = prediction.arguments.get("practitioner_name")
        assert spoken and "Rey" in spoken
        assert prediction.arguments.get("preferred_date") == "mardi"
