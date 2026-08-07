"""A15 et A16 — choisir la source de chaque argument, puis apprendre le reste.

Ce que la mesure d'A13 et A14 a montre, argument par argument : DIET est
nettement meilleur sur les noms de praticien et sur les heures, les regles sont
nettement meilleures sur les dates. Une composition qui prend l'un ou l'autre en
bloc laisse donc de la precision des deux cotes.

**A15** arbitre argument par argument. La preference n'est pas ecrite a la main
— ce serait s'ajuster sur ce que le test a revele — elle se mesure sur la
**validation**, jamais vue par DIET ni par le classifieur, tous deux entraines
sur `train`. Chaque source est appliquee aux enonces de validation avec leur
fonction attendue, et l'argument revient a celle qui a eu raison le plus
souvent. Si la source retenue ne trouve rien, l'autre est consultee : un
argument absent coute plus cher que le detour.

**A16** ajoute la seule chose qu'aucune des deux sources ne sait faire. Les
arguments enumeres — `topic`, `reason`, `reason_category` — ne figurent pas
litteralement dans la phrase : DIET ne peut pas les annoter et les regles se
contentent d'une valeur par defaut. Sur le corpus, cette valeur par defaut n'est
jamais la bonne pour `topic` : quatorze sujets possibles, environ trois cents
cas, zero pour cent. Un petit classifieur lexical entraine sur `train` — le meme
corpus que tout le monde — repond a la question que ni l'extraction de spans ni
une constante ne peuvent traiter.
"""

from __future__ import annotations

from typing import Any

from ivr_bench.domain.models import ToolDefinition
from ivr_bench.generators.corpus import load_split
from ivr_bench.metrics.calls import normalise_argument
from ivr_bench.routers.composite.router import ClassifierDietRouter
from ivr_bench.routers.diet.dataset import ENUMERATED
from ivr_bench.routers.registry import register

DIET = "diet"
RULES = "rules"


class ArbitratedRouter(ClassifierDietRouter):
    """A15 — la source de chaque argument est decidee sur la validation."""

    name = "classifier_diet_arbitrated"

    #: Les arguments enumeres restent deduits de la fonction, comme chez A13.
    learn_enumerations = False

    def __init__(
        self,
        encoder: str = "embeddinggemma_128",
        strategy: str = "kmeans",
        top_k_prototypes: int = 30,
        train_split: str = "train",
        validation_split: str = "validation",
        seed: int = 42,
        model_path: str | None = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            strategy=strategy,
            top_k_prototypes=top_k_prototypes,
            train_split=train_split,
            seed=seed,
            model_path=model_path,
        )
        self._validation_split = validation_split
        self._preference: dict[str, str] = {}
        self._enumerations: dict[tuple[str, str], Any] = {}
        if self.learn_enumerations:
            self._fit_enumerations(train_split, seed)

    # -- construction -------------------------------------------------------

    def _fit_enumerations(self, train_split: str, seed: int) -> None:
        """Un classifieur par argument enumere, entraine sur `train` seul."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for case in load_split(train_split):
            function = case.expected.tool_name
            for name, value in case.expected.arguments.items():
                if name not in ENUMERATED or not isinstance(value, str) or not value:
                    continue
                grouped.setdefault((function, name), []).append((case.utterance, value))

        for key, examples in grouped.items():
            texts = [text for text, _ in examples]
            labels = [label for _, label in examples]
            if len(set(labels)) < 2:
                # Une seule valeur observee : un modele n'apprendrait rien de
                # plus qu'une constante, et il faut le dire ainsi.
                self._enumerations[key] = labels[0]
                continue
            model = make_pipeline(
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            )
            model.fit(texts, labels)
            self._enumerations[key] = model

    def prepare(self, utterances: list[str]) -> None:
        """Analyse DIET du lot de mesure **et** de la validation.

        Les deux passent dans le meme lot : l'arbitrage a besoin des entites de
        DIET sur la validation, et une seconde ouverture du sidecar couterait un
        second chargement du modele.
        """
        cases = load_split(self._validation_split)
        super().prepare([*utterances, *(case.utterance for case in cases)])
        self._preference = self._arbitrate(cases)

    def _arbitrate(self, cases: list[Any]) -> dict[str, str]:
        """Compte, argument par argument, laquelle des deux sources a raison."""
        scores: dict[str, dict[str, int]] = {}
        for case in cases:
            definition = self._catalog.get(case.expected.tool_name)
            if not definition.executable:
                continue
            parsed = self._diet.parsed(case.utterance)
            from_diet, _ = self._from_entities(definition, parsed.get("entities", []))
            from_rules = self._arguments(definition.name, case.utterance)

            for name, expected in case.expected.arguments.items():
                if name in ENUMERATED:
                    continue
                want = normalise_argument(expected)
                tally = scores.setdefault(name, {DIET: 0, RULES: 0})
                tally[DIET] += int(normalise_argument(from_diet.get(name)) == want)
                tally[RULES] += int(normalise_argument(from_rules.get(name)) == want)

        # A egalite, les regles l'emportent : elles n'ont besoin d'aucun modele.
        return {
            name: DIET if tally[DIET] > tally[RULES] else RULES for name, tally in scores.items()
        }

    # -- inference ----------------------------------------------------------

    def _compose(
        self,
        definition: ToolDefinition,
        utterance: str,
        extractable: bool,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not definition.executable:
            return {}

        from_diet = (
            self._diet_arguments(definition, utterance, metadata)
            if extractable
            else dict.fromkeys(definition.parameter_names)
        )
        from_rules = self._arguments(definition.name, utterance) if extractable else {}

        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for name in definition.parameter_names:
            if name in ENUMERATED:
                values[name] = self._enumerated(definition, name, utterance)
                sources[name] = "learned" if self.learn_enumerations else "definition"
                continue
            preferred = self._preference.get(name, RULES)
            first = from_diet if preferred == DIET else from_rules
            second = from_rules if preferred == DIET else from_diet
            if first.get(name) is not None:
                values[name] = first[name]
                sources[name] = preferred
            else:
                values[name] = second.get(name)
                sources[name] = (RULES if preferred == DIET else DIET) if values[name] else "aucune"

        metadata["sources"] = sources
        return values

    def _enumerated(self, definition: ToolDefinition, name: str, utterance: str) -> Any:
        """Valeur d'un argument enumere : apprise (A16) ou deduite (A15)."""
        parameter = definition.parameter(name)
        if parameter is None or not parameter.enum:
            return None
        model = self._enumerations.get((definition.name, name))
        if model is None:
            return parameter.enum[-1]
        if isinstance(model, str):
            return model
        return str(model.predict([utterance])[0])


class ArbitratedEnumRouter(ArbitratedRouter):
    """A16 — meme arbitrage, plus un classifieur par argument enumere."""

    name = "classifier_diet_enum"
    learn_enumerations = True


register("classifier_diet_arbitrated")(ArbitratedRouter)
register("classifier_diet_enum")(ArbitratedEnumRouter)
