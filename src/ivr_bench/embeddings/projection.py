"""Reduction de dimension par projection apprise (§10.4).

64 dimensions ne fait pas partie des sorties prevues par EmbeddingGemma. La
specification interdit de tronquer un embedding qui n'a pas ete entraine pour
l'etre : on apprend donc une projection, **sur les seules donnees
d'entrainement**. Ajuster la reduction sur le test reviendrait a laisser le
banc d'essai regarder ses propres reponses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ivr_bench.embeddings.base import EmbeddingEncoder, Vectors, l2_normalize


class LearnedProjection:
    """Projection lineaire apprise par analyse en composantes principales."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._mean: Vectors | None = None
        self._components: Vectors | None = None

    @property
    def is_fitted(self) -> bool:
        return self._components is not None

    def fit(self, vectors: Vectors) -> LearnedProjection:
        """Ajuste la projection sur des vecteurs d'entrainement uniquement."""
        if vectors.ndim != 2:
            raise ValueError("matrice de vecteurs attendue")
        if vectors.shape[0] < self.dimension:
            raise ValueError(
                f"{vectors.shape[0]} exemples pour {self.dimension} dimensions : "
                "trop peu pour ajuster une projection honnete"
            )

        mean = vectors.mean(axis=0, keepdims=True).astype(np.float32)
        centred = vectors - mean
        # SVD plutot que la matrice de covariance : plus stable numeriquement sur
        # des vecteurs deja normalises.
        _, _, right = np.linalg.svd(centred, full_matrices=False)
        self._mean = mean
        self._components = np.ascontiguousarray(right[: self.dimension].T.astype(np.float32))
        return self

    def transform(self, vectors: Vectors) -> Vectors:
        if self._components is None or self._mean is None:
            raise RuntimeError("projection non ajustee : appelez fit() d'abord")
        projected = (vectors - self._mean) @ self._components
        # Renormalisation : la similarite cosinus reste un produit scalaire.
        return l2_normalize(projected.astype(np.float32))

    def save(self, path: Path) -> None:
        if self._components is None or self._mean is None:
            raise RuntimeError("projection non ajustee")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mean=self._mean, components=self._components)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "dimension": self.dimension,
                    "source_dimension": int(self._mean.shape[1]),
                }
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> LearnedProjection:
        payload = np.load(path)
        projection = cls(int(payload["components"].shape[1]))
        projection._mean = payload["mean"].astype(np.float32)
        projection._components = payload["components"].astype(np.float32)
        return projection


class ProjectedEncoder:
    """Encodeur de base suivi d'une projection apprise."""

    def __init__(self, base: EmbeddingEncoder, projection: LearnedProjection, name: str) -> None:
        self._base = base
        self._projection = projection
        self.name = name
        self.dimension = projection.dimension

    def encode(self, texts: list[str], batch_size: int = 32) -> Vectors:
        encode = self._base.encode
        vectors: Vectors = encode(texts, batch_size=batch_size)
        return self._projection.transform(vectors)
