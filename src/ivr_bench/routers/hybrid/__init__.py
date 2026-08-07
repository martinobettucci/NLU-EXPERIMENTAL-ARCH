"""Architectures a preselection semantique."""

from ivr_bench.routers.hybrid.backbone import RetrievalBackbone
from ivr_bench.routers.hybrid.embedding_only import EmbeddingOnlyRouter

__all__ = ["EmbeddingOnlyRouter", "RetrievalBackbone"]
