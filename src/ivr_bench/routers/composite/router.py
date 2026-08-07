"""A13 et A14 — separer le choix de la fonction de l'extraction des arguments.

Les campagnes precedentes donnent deux resultats qui ne pointent pas dans la
meme direction : le classifieur sur embeddings (A9) choisit mieux la fonction
que DIET, tandis que DIET est, parmi les modeles appris, celui qui extrait le
mieux les arguments. Rien n'oblige a confier les deux decisions au meme modele.

Ces deux architectures les separent :

- **A13** la fonction vient du classifieur, les arguments des entites de DIET ;
- **A14** identique, mais les arguments que DIET n'a pas trouves sont completes
  par l'extracteur par regles deja utilise par A0, A5 et A9.

La comparaison avec A9 est exactement appariee : meme encodeur, meme corpus,
meme graine, donc les **memes fonctions predites**. L'ecart d'exactitude
d'arguments porte alors sur l'extracteur et sur rien d'autre — ce que la
comparaison A0 contre A9 ne permettait pas, puisque l'exactitude d'arguments ne
se compte que sur les cas ou la fonction est correcte, et que ces cas ne sont
pas les memes d'une architecture a l'autre.

Deux points de mise en oeuvre qui ne sont pas cosmetiques.

**Realignement des roles.** Les entites de DIET portent le nom de l'argument de
*sa* fonction : « mardi » est annote `preferred_date` sous
`request_new_appointment` et `date_from` sous `list_appointments`. Quand le
classifieur retient une autre fonction que l'intention de DIET, une
correspondance par nom strict jetterait une valeur pourtant correctement
localisee. Les roles ci-dessous realignent ces noms de facon deterministe, une
seule valeur par role, ce qui evite de remplir un second champ date que rien
dans la phrase ne justifie.

**DIET n'est appele que s'il a quelque chose a extraire.** Une fonction dont
tous les arguments sont enumeres — `emergency_handoff`, `human_handoff`,
`answer_general_information` — ou qui n'en a aucun ne tire rien d'une analyse
d'entites : la composition s'arrete au classifieur. Ce n'est pas une economie
de banc d'essai, c'est le comportement qu'aurait le systeme deploye, et la
latence publiee doit le refleter.
"""

from __future__ import annotations

import time
from typing import Any

from ivr_bench.domain.models import RouterPrediction, SessionContext, ToolDefinition
from ivr_bench.routers.classifier.router import EmbeddingClassifierRouter, RuleArgumentMixin
from ivr_bench.routers.diet.dataset import ENUMERATED
from ivr_bench.routers.diet.router import DietRouter
from ivr_bench.routers.registry import register

# Arguments qui designent la meme chose sous des noms differents selon la
# fonction. L'ordre a l'interieur d'un role est sans importance : il dit
# seulement quels noms sont interchangeables.
ROLES: tuple[frozenset[str], ...] = (
    frozenset({"date_from", "date_to", "preferred_date", "preferred_new_date", "current_date"}),
    frozenset({"preferred_time", "preferred_new_time"}),
)


def _role_of(name: str) -> frozenset[str] | None:
    for role in ROLES:
        if name in role:
            return role
    return None


class ClassifierDietRouter(RuleArgumentMixin):
    """A13 — fonction choisie par le classifieur, arguments extraits par DIET."""

    name = "classifier_diet"

    #: Un argument que DIET n'a pas trouve reste absent.
    fill_with_rules = False

    def __init__(
        self,
        encoder: str = "embeddinggemma_128",
        strategy: str = "kmeans",
        top_k_prototypes: int = 30,
        train_split: str = "train",
        seed: int = 42,
        model_path: str | None = None,
    ) -> None:
        super().__init__()
        self._classifier = EmbeddingClassifierRouter(
            encoder=encoder,
            strategy=strategy,
            top_k_prototypes=top_k_prototypes,
            train_split=train_split,
            seed=seed,
        )
        self._diet = DietRouter(model_path)

    def prepare(self, utterances: list[str]) -> None:
        """Analyse DIET du lot entier, modele charge une seule fois."""
        self._diet.prepare(utterances)

    def _from_entities(
        self, definition: ToolDefinition, entities: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], int]:
        """Arguments issus des entites de DIET, roles realignes.

        Renvoie aussi le nombre de valeurs recuperees par realignement : c'est
        la part que la composition aurait perdue avec une correspondance par
        nom strict, et elle merite d'etre lisible dans les traces.
        """
        values: dict[str, Any] = dict.fromkeys(definition.parameter_names)
        remaining: list[tuple[str, Any]] = []

        for entity in entities:
            name = entity.get("entity")
            value = entity.get("value")
            if not isinstance(name, str) or value is None:
                continue
            if name in values and values[name] is None:
                values[name] = value
            else:
                remaining.append((name, value))

        realigned = 0
        for name, value in remaining:
            role = _role_of(name)
            if role is None:
                continue
            # Un role deja renseigne ne recoit pas de seconde valeur : deux
            # dates la ou la phrase n'en exprime qu'une seraient une invention,
            # et le §12 la fait payer plus cher qu'un manque.
            if any(values.get(parameter) is not None for parameter in role if parameter in values):
                continue
            target = next(
                (
                    parameter
                    for parameter in definition.parameter_names
                    if parameter in role and values[parameter] is None
                ),
                None,
            )
            if target is not None:
                values[target] = value
                realigned += 1

        # Les arguments enumeres ne figurent pas litteralement dans la phrase :
        # ils se deduisent de la fonction, comme chez les deux parents.
        for name in ENUMERATED:
            parameter = definition.parameter(name)
            if name in values and values[name] is None and parameter and parameter.enum:
                values[name] = parameter.enum[-1]
        return values, realigned

    def _diet_arguments(
        self, definition: ToolDefinition, utterance: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Arguments vus par DIET, l'analyse n'etant demandee que si elle sert."""
        parsed = self._diet.parsed(utterance)
        values, realigned = self._from_entities(definition, parsed.get("entities", []))
        metadata["diet_ms"] = round(float(parsed.get("latency_ms", 0.0)), 3)
        metadata["diet_intent"] = (parsed.get("intent") or {}).get("name")
        metadata["realigned"] = realigned
        return values

    def _compose(
        self,
        definition: ToolDefinition,
        utterance: str,
        extractable: bool,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble les arguments. Point de variation entre A13, A14 et A15."""
        if not definition.executable:
            return {}
        if not extractable:
            # Aucun argument litteral a chercher : la composition s'arrete au
            # classifieur, DIET n'aurait rien a dire.
            values, _ = self._from_entities(definition, [])
            return values

        values = self._diet_arguments(definition, utterance, metadata)
        if self.fill_with_rules:
            fallback = self._arguments(definition.name, utterance)
            for name, value in values.items():
                if value is None and fallback.get(name) is not None:
                    values[name] = fallback[name]
                    metadata["filled_by_rules"] += 1
        return values

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        selection = self._classifier.predict(utterance, session, tools)
        function = selection.tool_name
        metadata: dict[str, Any] = {
            **selection.metadata,
            "router": self.name,
            "classifier_ms": round(selection.latency_ms, 3),
            "diet_ms": 0.0,
            "diet_intent": None,
            "realigned": 0,
            "filled_by_rules": 0,
        }
        if function is None:
            return RouterPrediction(
                tool_name=None,
                arguments={},
                confidence=selection.confidence,
                candidates=selection.candidates,
                raw_output=None,
                latency_ms=selection.latency_ms,
                metadata=metadata,
            )

        definition = self._catalog.get(function)
        extractable = definition.executable and any(
            name not in ENUMERATED for name in definition.parameter_names
        )

        started = time.perf_counter()
        arguments = self._compose(definition, utterance, extractable, metadata)
        merge_ms = (time.perf_counter() - started) * 1000.0
        return RouterPrediction(
            tool_name=function,
            arguments=arguments,
            confidence=selection.confidence,
            candidates=selection.candidates,
            raw_output=None,
            # Cout reel de la composition : le classifieur, puis l'analyse DIET
            # du meme enonce quand elle a lieu. Cette analyse est faite en lot
            # avant la mesure, mais son temps est celui du sidecar, releve au
            # plus pres du modele.
            latency_ms=selection.latency_ms + float(metadata["diet_ms"]) + merge_ms,
            metadata=metadata,
        )


class ClassifierDietRulesRouter(ClassifierDietRouter):
    """A14 — meme composition, arguments manquants completes par regles."""

    name = "classifier_diet_rules"
    fill_with_rules = True


register("classifier_diet")(ClassifierDietRouter)
register("classifier_diet_rules")(ClassifierDietRulesRouter)
