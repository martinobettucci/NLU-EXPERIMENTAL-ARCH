"""A3 et A7 — FunctionGemma zero-shot, puis precede du retriever (§9).

L'editeur presente ce modele comme une base destinee a etre specialisee. La
variante zero-shot est donc mesuree pour ce qu'elle est — un point de depart —
et publiee separement de la variante specialisee (A4), conformement au §9.A4.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import RouterPrediction, SessionContext, ToolCandidate, ToolDefinition
from ivr_bench.domain.validation import OutputValidator
from ivr_bench.routers.functiongemma import runtime
from ivr_bench.routers.hybrid.backbone import RetrievalBackbone
from ivr_bench.routers.registry import register


class FunctionGemmaRouter:
    """A3 — catalogue complet, sans specialisation."""

    name = "functiongemma_zero_shot"

    def __init__(self, max_new_tokens: int = 96) -> None:
        self._catalog = default_catalog()
        self._validator = OutputValidator(self._catalog)
        self._max_new_tokens = max_new_tokens

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        started = time.perf_counter()
        selection = tools or list(self._catalog.functions)
        outcome, result = self._call(utterance, selection)
        return RouterPrediction(
            tool_name=outcome.tool_name if outcome.is_usable else None,
            arguments=outcome.arguments if outcome.is_usable else {},
            confidence=None,
            raw_output=result.raw,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "router": self.name,
                "tools_offered": len(selection),
                "decode_ms": round(result.latency_ms, 2),
                "prompt_tokens": result.prompt_tokens,
                "validity": outcome.validity,
                "repairs": list(outcome.repairs),
            },
        )

    def _call(self, utterance: str, tools: list[ToolDefinition]) -> tuple[Any, runtime.GemmaCall]:
        result = runtime.call(
            utterance, runtime.tools_payload(tools), max_new_tokens=self._max_new_tokens
        )
        # Une sortie sans appel reconnaissable reste invalide : le modele a
        # repondu en prose au lieu d'appeler une fonction, et le benchmark doit
        # le compter comme tel plutot que de le rattraper.
        payload = json.dumps(result.parsed, ensure_ascii=False) if result.parsed else result.raw
        return self._validator.validate(payload), result


class HybridFunctionGemmaRouter(FunctionGemmaRouter):
    """A7 — retriever top 2, puis FunctionGemma sur ces seules fonctions."""

    name = "hybrid_functiongemma_top2"

    def __init__(
        self,
        encoder: str = "embeddinggemma_128",
        strategy: str = "kmeans",
        top_k_prototypes: int = 30,
        candidates: int = 2,
        max_new_tokens: int = 96,
    ) -> None:
        super().__init__(max_new_tokens=max_new_tokens)
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
        shortlist = [name for name, _ in ordered[: self._candidates]] or list(self._catalog.names)

        selection = list(self._catalog.subset(shortlist))
        outcome, result = self._call(utterance, selection)

        uncertainty = retrieval.ranking.uncertainty
        return RouterPrediction(
            tool_name=outcome.tool_name if outcome.is_usable else None,
            arguments=outcome.arguments if outcome.is_usable else {},
            confidence=round(ordered[0][1], 4) if ordered else None,
            candidates=tuple(
                ToolCandidate(tool_name=name, score=round(score, 4), rank=position)
                for position, (name, score) in enumerate(ordered[: len(shortlist)])
            ),
            raw_output=result.raw,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "router": self.name,
                "encoder": self._backbone.encoder_name,
                "tools_offered": len(selection),
                "encode_ms": round(retrieval.encode_ms, 3),
                "search_ms": round(retrieval.search_ms, 3),
                "decode_ms": round(result.latency_ms, 2),
                "prompt_tokens": result.prompt_tokens,
                "validity": outcome.validity,
                "repairs": list(outcome.repairs),
                "margin": round(uncertainty.margin, 4),
                "purity": round(uncertainty.purity, 4),
                "entropy": round(uncertainty.entropy, 4),
            },
        )


@register("functiongemma_zero_shot")
def _create_functiongemma_router(**kwargs: Any) -> FunctionGemmaRouter:
    return FunctionGemmaRouter(**kwargs)


@register("hybrid_functiongemma_top2")
def _create_hybrid_functiongemma_router(**kwargs: Any) -> HybridFunctionGemmaRouter:
    return HybridFunctionGemmaRouter(**kwargs)
