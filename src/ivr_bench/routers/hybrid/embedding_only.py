"""A5 — retriever semantique seul (§9).

La fonction la mieux classee est retenue directement, sans modele d'appel
d'outils. Les arguments sont extraits par les memes regles que la baseline : la
mesure porte ici sur la selection de fonction, pas sur l'extraction.

Cette baseline repond a une question precise : que vaut le retriever **avant**
qu'un micro-modele ne tranche ? L'ecart entre son rappel top 1 et son rappel
top 2 est la marge de manoeuvre que les architectures hybrides exploitent.
"""

from __future__ import annotations

import time
from typing import Any

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import RouterPrediction, SessionContext, ToolCandidate, ToolDefinition
from ivr_bench.generators.practitioners import load_name_banks
from ivr_bench.routers.hybrid.backbone import RetrievalBackbone
from ivr_bench.routers.registry import register
from ivr_bench.routers.rules import extraction


class EmbeddingOnlyRouter:
    """Selection directe de la fonction la mieux classee."""

    name = "embedding_only"

    def __init__(
        self,
        encoder: str = "embeddinggemma_128",
        strategy: str = "kmeans",
        top_k_prototypes: int = 30,
        minimum_score: float = 0.0,
    ) -> None:
        self._backbone = RetrievalBackbone(encoder, strategy, top_k_prototypes)
        self._catalog = default_catalog()
        self._specialties = list(load_name_banks()["specialties"])
        # Seuil de rejet : sous ce score, aucune fonction n'est retenue. A zero,
        # le routeur classe toujours, ce qui est le comportement de reference.
        self._minimum_score = minimum_score

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        started = time.perf_counter()
        result = self._backbone.retrieve(utterance)
        allowed = {tool.name for tool in tools} or set(self._catalog.names)

        ordered = [(name, score) for name, score in result.ranking.ordered if name in allowed]
        candidates = tuple(
            ToolCandidate(tool_name=name, score=round(score, 4), rank=position)
            for position, (name, score) in enumerate(ordered[:3])
        )

        if not ordered or ordered[0][1] < self._minimum_score:
            selected: str | None = "no_tool" if "no_tool" in allowed else None
            arguments: dict[str, Any] = {}
        else:
            selected = ordered[0][0]
            arguments = self._arguments(selected, utterance)

        uncertainty = result.ranking.uncertainty
        return RouterPrediction(
            tool_name=selected,
            arguments=arguments,
            confidence=round(ordered[0][1], 4) if ordered else None,
            candidates=candidates,
            raw_output=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "router": self.name,
                "encoder": self._backbone.encoder_name,
                "encode_ms": round(result.encode_ms, 3),
                "search_ms": round(result.search_ms, 3),
                # Signaux d'incertitude du §11, conserves pour l'analyse d'erreurs.
                "margin": round(uncertainty.margin, 4),
                "purity": round(uncertainty.purity, 4),
                "entropy": round(uncertainty.entropy, 4),
            },
        )

    def _arguments(self, function: str, utterance: str) -> dict[str, Any]:
        """Arguments extraits par regles, jamais devines."""
        definition = self._catalog.get(function)
        if not definition.executable:
            return {}

        values: dict[str, Any] = dict.fromkeys(definition.parameter_names)
        practitioner = extraction.extract_practitioner(utterance)
        date = extraction.extract_date(utterance)
        moment = extraction.extract_time(utterance)

        if "practitioner_name" in values:
            values["practitioner_name"] = practitioner
        if "specialty" in values:
            values["specialty"] = extraction.extract_specialty(utterance, self._specialties)
        if "preferred_date" in values:
            values["preferred_date"] = date
        if "preferred_new_date" in values:
            values["preferred_new_date"] = date
        if "date_from" in values:
            values["date_from"] = date
        if "preferred_time" in values:
            values["preferred_time"] = moment
        if "preferred_new_time" in values:
            values["preferred_new_time"] = moment

        # Les arguments enumeres ne s'extraient pas d'un enonce libre : les
        # laisser vides serait invalide, les deviner serait pire. Le retriever
        # n'est pas outille pour cette fonction, et le benchmark doit le voir.
        for name in ("topic", "reason", "reason_category"):
            if name in values and values[name] is None:
                parameter = definition.parameter(name)
                if parameter is not None and parameter.enum:
                    values[name] = parameter.enum[-1]
        return values


@register("embedding_only")
def _create_embedding_only_router(**kwargs: Any) -> EmbeddingOnlyRouter:
    return EmbeddingOnlyRouter(**kwargs)
