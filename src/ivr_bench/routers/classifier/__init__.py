"""Strategies entrainees : classifieur dense, classifieur lexical, k plus proches voisins."""

from ivr_bench.routers.classifier.router import (
    EmbeddingClassifierRouter,
    LexicalClassifierRouter,
    NearestNeighbourRouter,
)

__all__ = [
    "EmbeddingClassifierRouter",
    "LexicalClassifierRouter",
    "NearestNeighbourRouter",
]
