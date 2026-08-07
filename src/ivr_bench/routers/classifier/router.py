"""Strategies entrainees sur une seule phrase (§9, extension).

Trois approches que le catalogue initial n'avait pas, et qui sont pourtant les
plus repandues en production pour transformer une phrase en fonction :

- **A9** un classifieur lineaire sur les memes embeddings que le retriever ;
- **A10** un modele lexical TF-IDF, sans aucun reseau de neurones ;
- **A11** les k plus proches voisins sur l'index existant.

Elles partagent le corpus d'entrainement des autres architectures et le meme
extracteur d'arguments par regles, si bien que l'ecart mesure porte sur la
selection de fonction et sur rien d'autre.

Le classifieur lineaire repond a une question que le banc d'essai laissait
ouverte : le retriever a-t-il besoin d'un modele d'appel d'outils en aval, ou
suffit-il de lui apprendre la frontiere entre fonctions ?
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import RouterPrediction, SessionContext, ToolCandidate, ToolDefinition
from ivr_bench.generators.corpus import load_split
from ivr_bench.generators.practitioners import load_name_banks
from ivr_bench.routers.hybrid.backbone import RetrievalBackbone
from ivr_bench.routers.registry import register
from ivr_bench.routers.rules import extraction


class RuleArgumentMixin:
    """Extraction d'arguments commune, identique a celle du retriever seul."""

    def __init__(self) -> None:
        self._catalog = default_catalog()
        self._specialties = list(load_name_banks()["specialties"])

    def _arguments(self, function: str, utterance: str) -> dict[str, Any]:
        definition = self._catalog.get(function)
        if not definition.executable:
            return {}

        values: dict[str, Any] = dict.fromkeys(definition.parameter_names)
        date = extraction.extract_date(utterance)
        moment = extraction.extract_time(utterance)
        for key, value in (
            ("practitioner_name", extraction.extract_practitioner(utterance)),
            ("specialty", extraction.extract_specialty(utterance, self._specialties)),
            ("preferred_date", date),
            ("preferred_new_date", date),
            ("date_from", date),
            ("preferred_time", moment),
            ("preferred_new_time", moment),
        ):
            if key in values:
                values[key] = value

        # Les arguments enumeres ne s'extraient pas d'un enonce libre.
        for name in ("topic", "reason", "reason_category"):
            parameter = definition.parameter(name)
            if name in values and values[name] is None and parameter and parameter.enum:
                values[name] = parameter.enum[-1]
        return values


class EmbeddingClassifierRouter(RuleArgumentMixin):
    """A9 — regression logistique sur les embeddings du retriever.

    Le retriever compare une phrase a des prototypes ; ce classifieur apprend
    au contraire la frontiere entre fonctions. C'est la difference entre
    « de quoi cette phrase est-elle proche ? » et « qu'est-ce qui separe ces
    fonctions ? », et rien ne garantit a l'avance que la premiere question soit
    la bonne.
    """

    name = "embedding_classifier"

    def __init__(
        self,
        encoder: str = "embeddinggemma_128",
        strategy: str = "kmeans",
        top_k_prototypes: int = 30,
        train_split: str = "train",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._backbone = RetrievalBackbone(encoder, strategy, top_k_prototypes)
        self._encoder_name = encoder

        from sklearn.linear_model import LogisticRegression

        cases = load_split(train_split)
        from ivr_bench.embeddings.encoders import create_encoder

        model = create_encoder(encoder)
        vectors = model.encode([case.utterance for case in cases], batch_size=32)
        labels = [case.expected.tool_name for case in cases]

        self._model = LogisticRegression(
            max_iter=2000,
            # Les fonctions n'ont pas toutes le meme effectif : sans ce
            # reequilibrage, les classes minoritaires seraient sacrifiees.
            class_weight="balanced",
            random_state=seed,
        )
        self._model.fit(vectors, labels)
        self._classes = list(self._model.classes_)

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        started = time.perf_counter()
        allowed = {tool.name for tool in tools} or set(self._catalog.names)

        encode_started = time.perf_counter()
        vector = self._backbone.encode(utterance)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        probabilities = self._model.predict_proba(vector.reshape(1, -1))[0]
        ranked = sorted(
            (
                (name, float(score))
                for name, score in zip(self._classes, probabilities, strict=True)
                if name in allowed
            ),
            key=lambda item: (-item[1], item[0]),
        )
        selected, confidence = ranked[0] if ranked else (None, None)

        return RouterPrediction(
            tool_name=selected,
            arguments=self._arguments(selected, utterance) if selected else {},
            confidence=round(confidence, 4) if confidence is not None else None,
            candidates=tuple(
                ToolCandidate(tool_name=name, score=round(score, 4), rank=position)
                for position, (name, score) in enumerate(ranked[:3])
            ),
            raw_output=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "router": self.name,
                "encoder": self._encoder_name,
                "encode_ms": round(encode_ms, 3),
            },
        )


class LexicalClassifierRouter(RuleArgumentMixin):
    """A10 — TF-IDF et regression logistique, sans aucun embedding.

    La question est directe : que reste-t-il du gain des representations denses
    quand un modele lexical voit le meme corpus ? S'il tient, le cout d'un
    encodeur de 300 millions de parametres demande a etre justifie.
    """

    name = "lexical_classifier"

    def __init__(self, train_split: str = "train", seed: int = 42) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        cases = load_split(train_split)
        self._model = make_pipeline(
            # Caracteres autant que mots : les fautes de frappe et les erreurs de
            # transcription du corpus cassent les mots, pas les trigrammes.
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        self._model.fit(
            [case.utterance for case in cases], [case.expected.tool_name for case in cases]
        )
        self._classes = list(self._model.classes_)

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        started = time.perf_counter()
        allowed = {tool.name for tool in tools} or set(self._catalog.names)

        probabilities = self._model.predict_proba([utterance])[0]
        ranked = sorted(
            (
                (name, float(score))
                for name, score in zip(self._classes, probabilities, strict=True)
                if name in allowed
            ),
            key=lambda item: (-item[1], item[0]),
        )
        selected, confidence = ranked[0] if ranked else (None, None)

        return RouterPrediction(
            tool_name=selected,
            arguments=self._arguments(selected, utterance) if selected else {},
            confidence=round(confidence, 4) if confidence is not None else None,
            candidates=tuple(
                ToolCandidate(tool_name=name, score=round(score, 4), rank=position)
                for position, (name, score) in enumerate(ranked[:3])
            ),
            raw_output=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"router": self.name},
        )


class NearestNeighbourRouter(RuleArgumentMixin):
    """A11 — vote des k plus proches enonces d'entrainement.

    Variante du retriever qui n'agrege pas des prototypes mais vote parmi les
    voisins bruts : elle isole l'effet du resume en prototypes du §10.6.
    """

    name = "nearest_neighbour"

    def __init__(
        self,
        encoder: str = "embeddinggemma_128",
        strategy: str = "kmeans",
        top_k_prototypes: int = 30,
        neighbours: int = 15,
        train_split: str = "index",
    ) -> None:
        super().__init__()
        self._backbone = RetrievalBackbone(encoder, strategy, top_k_prototypes)
        self._neighbours = neighbours
        self._encoder_name = encoder

        from ivr_bench.embeddings.encoders import create_encoder

        cases = load_split(train_split)
        model = create_encoder(encoder)
        self._vectors = model.encode([case.utterance for case in cases], batch_size=32)
        self._labels = [case.expected.tool_name for case in cases]

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        started = time.perf_counter()
        allowed = {tool.name for tool in tools} or set(self._catalog.names)

        vector = self._backbone.encode(utterance)
        similarities = self._vectors @ vector
        top = np.argsort(-similarities)[: self._neighbours]

        # Vote pondere par la similarite : un voisin tres proche pese plus qu'un
        # voisin admis de justesse.
        weights: dict[str, float] = {}
        for position in top:
            label = self._labels[int(position)]
            if label in allowed:
                weights[label] = weights.get(label, 0.0) + float(similarities[position])

        ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        total = sum(weights.values()) or 1.0
        selected = ranked[0][0] if ranked else None

        return RouterPrediction(
            tool_name=selected,
            arguments=self._arguments(selected, utterance) if selected else {},
            confidence=round(ranked[0][1] / total, 4) if ranked else None,
            candidates=tuple(
                ToolCandidate(tool_name=name, score=round(score / total, 4), rank=position)
                for position, (name, score) in enumerate(ranked[:3])
            ),
            raw_output=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"router": self.name, "encoder": self._encoder_name},
        )


register("embedding_classifier")(EmbeddingClassifierRouter)
register("lexical_classifier")(LexicalClassifierRouter)
register("nearest_neighbour")(NearestNeighbourRouter)
