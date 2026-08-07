"""Index semantique des formulations hypothetiques.

Pour un catalogue de cette taille, le produit matriciel exact est la reference
(§11) : il est exact, tient en memoire et se mesure sans ambiguite. Un index
approche ne se justifie qu'en variante, et devra prouver qu'il gagne quelque
chose.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ivr_bench.embeddings.base import Vectors
from ivr_bench.generators.utterances import GeneratedCase
from ivr_bench.retrieval import prototypes as prototype_builders
from ivr_bench.retrieval.aggregation import Neighbour, Ranking, aggregate
from ivr_bench.retrieval.prototypes import PrototypeStrategy


@dataclass(frozen=True)
class IndexMetadata:
    """Tout ce qu'il faut pour rejouer la construction a l'identique."""

    encoder: str
    dimension: int
    strategy: str
    prototypes_per_function: int
    seed: int
    source_cases: int
    built_at: str
    content_sha256: str


class SemanticIndex:
    """Prototypes par fonction et recherche exacte."""

    def __init__(
        self,
        vectors: Vectors,
        owners: list[str],
        metadata: IndexMetadata,
    ) -> None:
        if vectors.shape[0] != len(owners):
            raise ValueError("un proprietaire est attendu par prototype")
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._owners = owners
        self.metadata = metadata

    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    @property
    def functions(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._owners)))

    def search(self, query: Vectors, top_k: int = 30) -> list[Neighbour]:
        """Voisins les plus proches, similarite cosinus exacte."""
        if query.ndim == 2:
            query = query[0]
        similarities = self._vectors @ query
        count = min(top_k, similarities.shape[0])
        # argpartition puis tri du seul segment retenu : inutile de trier tout
        # l'index pour n'en garder que trente elements.
        selected = np.argpartition(-similarities, count - 1)[:count]
        selected = selected[np.argsort(-similarities[selected])]
        return [
            Neighbour(
                function=self._owners[int(position)],
                similarity=float(similarities[position]),
            )
            for position in selected
        ]

    def rank(self, query: Vectors, top_k: int = 30) -> Ranking:
        """Recherche puis agregation par fonction."""
        return aggregate(self.search(query, top_k=top_k))

    # -- persistance --------------------------------------------------------

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "index.npz"
        np.savez_compressed(target, vectors=self._vectors, owners=np.array(self._owners))
        (directory / "index.manifest.json").write_text(
            json.dumps(self.metadata.__dict__, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, directory: Path) -> SemanticIndex:
        payload = np.load(directory / "index.npz", allow_pickle=False)
        raw = json.loads((directory / "index.manifest.json").read_text(encoding="utf-8"))
        return cls(
            vectors=payload["vectors"].astype(np.float32),
            owners=[str(item) for item in payload["owners"]],
            metadata=IndexMetadata(**raw),
        )


def build_index(
    cases: list[GeneratedCase],
    encoder: Any,
    strategy: PrototypeStrategy = "kmeans",
    prototypes_per_function: int = 8,
    seed: int = 42,
    batch_size: int = 32,
) -> SemanticIndex:
    """Encode les formulations et resume chaque fonction en prototypes."""
    if not cases:
        raise ValueError("aucun enonce a indexer")

    by_function: dict[str, list[str]] = {}
    for case in cases:
        by_function.setdefault(case.expected.tool_name, []).append(case.utterance)

    vectors: list[Vectors] = []
    owners: list[str] = []
    for function in sorted(by_function):
        encoded = encoder.encode(by_function[function], batch_size=batch_size)
        summary = prototype_builders.build(
            encoded, strategy=strategy, count=prototypes_per_function, seed=seed
        )
        vectors.append(summary)
        owners.extend([function] * summary.shape[0])

    stacked = np.vstack(vectors).astype(np.float32)
    digest = hashlib.sha256(stacked.tobytes()).hexdigest()
    metadata = IndexMetadata(
        encoder=encoder.name,
        dimension=int(stacked.shape[1]),
        strategy=strategy,
        prototypes_per_function=prototypes_per_function,
        seed=seed,
        source_cases=len(cases),
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        content_sha256=digest,
    )
    return SemanticIndex(stacked, owners, metadata)
