"""A2, A6 et A8 — Needle seul, puis precede du retriever (§9).

A2 recoit le catalogue complet. A6 ne lui transmet que les deux fonctions
retenues par le retriever. A8 fait varier ce nombre selon l'incertitude.

Dans tous les cas, c'est **l'enonce original** qui est transmis, jamais une
formulation synthetique voisine : ajouter ces voisins au prompt est une ablation
du §27.9, pas le comportement par defaut.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import RouterPrediction, SessionContext, ToolCandidate, ToolDefinition
from ivr_bench.domain.validation import OutputValidator
from ivr_bench.routers.hybrid.backbone import RetrievalBackbone
from ivr_bench.routers.needle import runtime
from ivr_bench.routers.registry import register


def _unwrap(raw: str) -> str:
    """Deballe l'enveloppe de sortie documentee de Needle.

    Le modele renvoie une liste d'appels : c'est son format officiel, pas une
    sortie malformee. Extraire l'appel unique releve donc de l'adaptateur, pas
    de la reparation du §24 — sans cela, chaque sortie correcte serait
    comptabilisee comme « reparee » et le taux de validite native du modele
    serait artificiellement nul.
    """
    text = raw.strip()
    if not text.startswith("["):
        return raw
    try:
        calls = json.loads(text)
    except json.JSONDecodeError:
        return raw
    if isinstance(calls, list) and not calls:
        # Liste vide : c'est ainsi que Needle dit « aucun outil applicable ».
        # Le catalogue nomme cette reponse `no_tool` (§5.7). La traduire releve
        # de l'adaptateur, pas de la reparation : la compter comme sortie
        # invalide punirait le modele pour avoir correctement refuse d'agir, et
        # rendrait son rappel `no_tool` structurellement nul.
        return json.dumps({"name": "no_tool", "arguments": {}}, ensure_ascii=False)
    if isinstance(calls, list) and len(calls) == 1:
        return json.dumps(calls[0], ensure_ascii=False)
    return raw


class NeedleRouter:
    """A2 — Needle sur le catalogue complet."""

    name = "needle_full"

    def __init__(self, max_gen_len: int = 128) -> None:
        self._catalog = default_catalog()
        self._validator = OutputValidator(self._catalog)
        self._max_gen_len = max_gen_len

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        started = time.perf_counter()
        selection = tools or list(self._catalog.functions)
        outcome, raw, result = self._call(utterance, selection)
        return RouterPrediction(
            tool_name=outcome.tool_name if outcome.is_usable else None,
            arguments=outcome.arguments if outcome.is_usable else {},
            confidence=None,
            raw_output=raw,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "router": self.name,
                "tools_offered": len(selection),
                "decode_ms": round(result.latency_ms, 2),
                "prompt_tokens": result.prompt_tokens,
                "encoder_window": result.encoder_window,
                # La validite est publiee separement des resultats principaux :
                # une sortie reparee n'est pas une sortie native (§24).
                "validity": outcome.validity,
                "repairs": list(outcome.repairs),
            },
        )

    def _call(
        self, utterance: str, tools: list[ToolDefinition]
    ) -> tuple[Any, str, runtime.NeedleCall]:
        payload = runtime.tools_payload(tools)
        result = runtime.call(utterance, payload, max_gen_len=self._max_gen_len)
        outcome = self._validator.validate(_unwrap(result.raw))
        return outcome, result.raw, result


class HybridNeedleRouter(NeedleRouter):
    """A6 — retriever top 2, puis Needle sur ces seules fonctions."""

    name = "hybrid_needle_top2"

    def __init__(
        self,
        encoder: str = "embeddinggemma_128",
        strategy: str = "kmeans",
        top_k_prototypes: int = 30,
        candidates: int = 2,
        max_gen_len: int = 128,
    ) -> None:
        super().__init__(max_gen_len=max_gen_len)
        self._backbone = RetrievalBackbone(encoder, strategy, top_k_prototypes)
        self._candidates = candidates

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        started = time.perf_counter()
        allowed = {tool.name for tool in tools} or set(self._catalog.names)

        retrieval = self._backbone.retrieve(utterance)
        ordered = [(name, score) for name, score in retrieval.ranking.ordered if name in allowed]
        shortlist = self._shortlist(ordered, retrieval.ranking.uncertainty)

        selection = list(self._catalog.subset(shortlist))
        outcome, raw, result = self._call(utterance, selection)

        uncertainty = retrieval.ranking.uncertainty
        return RouterPrediction(
            tool_name=outcome.tool_name if outcome.is_usable else None,
            arguments=outcome.arguments if outcome.is_usable else {},
            confidence=round(ordered[0][1], 4) if ordered else None,
            candidates=tuple(
                ToolCandidate(tool_name=name, score=round(score, 4), rank=position)
                for position, (name, score) in enumerate(ordered[: len(shortlist)])
            ),
            raw_output=raw,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "router": self.name,
                "encoder": self._backbone.encoder_name,
                "tools_offered": len(selection),
                "encode_ms": round(retrieval.encode_ms, 3),
                "search_ms": round(retrieval.search_ms, 3),
                "decode_ms": round(result.latency_ms, 2),
                "prompt_tokens": result.prompt_tokens,
                "encoder_window": result.encoder_window,
                "validity": outcome.validity,
                "repairs": list(outcome.repairs),
                "margin": round(uncertainty.margin, 4),
                "purity": round(uncertainty.purity, 4),
                "entropy": round(uncertainty.entropy, 4),
            },
        )

    def _shortlist(self, ordered: list[tuple[str, float]], uncertainty: Any) -> list[str]:
        """Nombre fixe de candidats : c'est la variante de reference."""
        del uncertainty
        return [name for name, _ in ordered[: self._candidates]] or list(self._catalog.names)


class AdaptiveHybridRouter(HybridNeedleRouter):
    """A8 — le nombre de candidats depend de l'incertitude (§9.A8)."""

    name = "hybrid_adaptive"

    def __init__(
        self,
        encoder: str = "embeddinggemma_128",
        strategy: str = "kmeans",
        top_k_prototypes: int = 30,
        confident_score: float = 0.75,
        confident_margin: float = 0.10,
        ambiguous_margin: float = 0.03,
        max_gen_len: int = 128,
    ) -> None:
        super().__init__(
            encoder=encoder,
            strategy=strategy,
            top_k_prototypes=top_k_prototypes,
            candidates=2,
            max_gen_len=max_gen_len,
        )
        self._confident_score = confident_score
        self._confident_margin = confident_margin
        self._ambiguous_margin = ambiguous_margin

    def _shortlist(self, ordered: list[tuple[str, float]], uncertainty: Any) -> list[str]:
        if not ordered:
            return list(self._catalog.names)

        # Score eleve ET marge nette : une seule fonction suffit, et l'appel
        # devient plus court. C'est la ou la variante adaptative gagne du temps.
        if uncertainty.top_score >= self._confident_score and (
            uncertainty.margin >= self._confident_margin
        ):
            return [ordered[0][0]]

        # Ambiguite marquee : on elargit plutot que de parier.
        if uncertainty.margin < self._ambiguous_margin:
            return [name for name, _ in ordered[:4]]

        return [name for name, _ in ordered[:2]]


@register("needle_full")
def _create_needle_router(**kwargs: Any) -> NeedleRouter:
    return NeedleRouter(**kwargs)


@register("hybrid_needle_top2")
def _create_hybrid_needle_router(**kwargs: Any) -> HybridNeedleRouter:
    return HybridNeedleRouter(**kwargs)


@register("hybrid_adaptive")
def _create_adaptive_router(**kwargs: Any) -> AdaptiveHybridRouter:
    return AdaptiveHybridRouter(**kwargs)
