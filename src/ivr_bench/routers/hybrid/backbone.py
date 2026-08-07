"""Partie commune aux architectures a preselection semantique.

Encoder l'enonce, interroger l'index, agreger par fonction : c'est ce que A5, A6,
A7 et A8 partagent. Ce qui les distingue est ce qu'elles font ensuite du
classement, pas la maniere de l'obtenir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ivr_bench.domain.paths import results_dir
from ivr_bench.embeddings.encoders import create_encoder
from ivr_bench.retrieval.aggregation import Ranking
from ivr_bench.retrieval.index import SemanticIndex


@dataclass(frozen=True)
class RetrievalResult:
    ranking: Ranking
    encode_ms: float
    search_ms: float


def index_directory(encoder: str, strategy: str) -> Path:
    return results_dir() / "index" / f"{encoder}_{strategy}"


class RetrievalBackbone:
    """Encodeur et index charges une seule fois, reutilises a chaque appel."""

    def __init__(
        self,
        encoder: str = "embeddinggemma_128",
        strategy: str = "kmeans",
        top_k_prototypes: int = 30,
    ) -> None:
        directory = index_directory(encoder, strategy)
        if not (directory / "index.npz").is_file():
            raise FileNotFoundError(f"index absent : {directory}. Lancez 'ivr-bench index build'.")
        self._encoder = create_encoder(encoder)
        self._index = SemanticIndex.load(directory)
        self._top_k = top_k_prototypes

        if self._index.metadata.dimension != self._encoder.dimension:
            # Un index construit avec une autre dimension produirait des scores
            # silencieusement absurdes plutot qu'une erreur.
            raise ValueError(
                f"index en {self._index.metadata.dimension}d contre encodeur en "
                f"{self._encoder.dimension}d : reconstruisez l'index."
            )

    @property
    def encoder_name(self) -> str:
        return self._encoder.name

    @property
    def index(self) -> SemanticIndex:
        return self._index

    def retrieve(self, utterance: str) -> RetrievalResult:
        """Classe les fonctions candidates, en mesurant chaque etape (§17.5)."""
        started = time.perf_counter()
        vector = self._encoder.encode([utterance], batch_size=1)[0]
        encoded_at = time.perf_counter()
        ranking = self._index.rank(vector, top_k=self._top_k)
        finished = time.perf_counter()
        return RetrievalResult(
            ranking=ranking,
            encode_ms=(encoded_at - started) * 1000.0,
            search_ms=(finished - encoded_at) * 1000.0,
        )
