"""A12 — la proposition d'origine, implementee telle qu'enoncee.

Le pipeline decrit est :

    intentions -> definitions de fonctions
    -> questions hypothetiques generees hors ligne (des centaines par intention)
    -> on ne conserve PAS les phrases, seulement leurs vecteurs, courts et rapides
    -> au runtime : similarite semantique, rerank par delta cosinus,
       on retient 2 definitions de fonctions
    -> Needle recoit la phrase initiale et ces deux definitions uniquement

Ce que cette architecture change par rapport a A6, et pourquoi elle existe
separement : A6 classe les fonctions avec le score compose du §11
(0,55 x meilleure similarite + 0,35 x moyenne des trois meilleurs + 0,10 x
soutien) et ne se sert du delta que comme signal d'incertitude — la
specification l'exige explicitement. La proposition d'origine, elle, classe sur
la **similarite maximale seule** et se sert du **delta cosinus** pour trancher.
Ce sont deux algorithmes differents ; les opposer dit lequel des deux a raison,
ce qu'aucune des deux ne peut affirmer seule.

Une ambiguite demeure dans l'enonce : « rerank par delta cosinus » peut se lire
comme un reordonnancement, ou comme un critere de coupe. C'est la seconde
lecture qui est implementee ici — le delta decide combien de definitions
partent au modele — parce que la premiere n'a pas de sens sur une liste deja
triee par similarite. Le nombre par defaut reste 2, comme demande.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import RouterPrediction, SessionContext, ToolCandidate, ToolDefinition
from ivr_bench.domain.validation import OutputValidator
from ivr_bench.routers.hybrid.backbone import RetrievalBackbone
from ivr_bench.routers.needle import runtime
from ivr_bench.routers.registry import register


class DeltaRerankRouter:
    """Similarite maximale, coupe par delta cosinus, puis Needle."""

    name = "hypothetical_delta_top2"

    def __init__(
        self,
        encoder: str = "embeddinggemma_128",
        strategy: str = "kmeans",
        top_k_prototypes: int = 30,
        candidates: int = 2,
        #: Au-dela de cet ecart avec le second, la premiere fonction part seule.
        decisive_delta: float = 0.12,
        max_gen_len: int = 128,
    ) -> None:
        self._catalog = default_catalog()
        self._validator = OutputValidator(self._catalog)
        self._backbone = RetrievalBackbone(encoder, strategy, top_k_prototypes)
        self._candidates = candidates
        self._decisive_delta = decisive_delta
        self._max_gen_len = max_gen_len

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        started = time.perf_counter()
        allowed = {tool.name for tool in tools} or set(self._catalog.names)

        retrieval = self._backbone.retrieve(utterance)

        # Classement sur la similarite maximale seule : pas de moyenne, pas de
        # soutien du voisinage. C'est le point de divergence avec A6.
        best: dict[str, float] = defaultdict(lambda: -1.0)
        for neighbour in retrieval.ranking.neighbours:
            if neighbour.function in allowed and neighbour.similarity > best[neighbour.function]:
                best[neighbour.function] = neighbour.similarity

        ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
        if not ordered:
            return self._empty(started, retrieval)

        delta = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else 1.0
        # Le delta decide de la largeur : franc, une seule definition suffit et
        # l'appel est plus court ; serre, on transmet les deux et on laisse le
        # modele trancher.
        keep = 1 if delta >= self._decisive_delta else self._candidates
        shortlist = [name for name, _ in ordered[:keep]]

        selection = list(self._catalog.subset(shortlist))
        payload = runtime.tools_payload(selection)
        result = runtime.call(utterance, payload, max_gen_len=self._max_gen_len)
        outcome = self._validator.validate(_unwrap(result.raw))

        return RouterPrediction(
            tool_name=outcome.tool_name if outcome.is_usable else None,
            arguments=outcome.arguments if outcome.is_usable else {},
            confidence=round(ordered[0][1], 4),
            candidates=tuple(
                ToolCandidate(tool_name=name, score=round(score, 4), rank=position)
                for position, (name, score) in enumerate(ordered[:3])
            ),
            raw_output=result.raw,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "router": self.name,
                "encoder": self._backbone.encoder_name,
                "tools_offered": len(selection),
                "cosine_delta": round(delta, 4),
                "encode_ms": round(retrieval.encode_ms, 3),
                "search_ms": round(retrieval.search_ms, 3),
                "decode_ms": round(result.latency_ms, 2),
                "prompt_tokens": result.prompt_tokens,
                "validity": outcome.validity,
            },
        )

    def _empty(self, started: float, retrieval: Any) -> RouterPrediction:
        return RouterPrediction(
            tool_name=None,
            arguments={},
            confidence=None,
            raw_output=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"router": self.name, "encode_ms": round(retrieval.encode_ms, 3)},
        )


def _unwrap(raw: str) -> str:
    from ivr_bench.routers.needle.router import _unwrap as unwrap_needle

    return unwrap_needle(raw)


register("hypothetical_delta_top2")(DeltaRerankRouter)
