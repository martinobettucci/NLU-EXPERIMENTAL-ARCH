"""Architectures de routage comparees.

L'import de ce paquet enregistre les architectures disponibles : un routeur
absent du registre serait invisible pour le banc d'essai.
"""

from ivr_bench.routers import rules  # noqa: F401
from ivr_bench.routers.base import Router
from ivr_bench.routers.classifier import router as classifier_router  # noqa: F401
from ivr_bench.routers.composite import router as composite_router  # noqa: F401
from ivr_bench.routers.diet import router as diet_router  # noqa: F401
from ivr_bench.routers.functiongemma import router as functiongemma_router  # noqa: F401
from ivr_bench.routers.hybrid import (
    delta_rerank,  # noqa: F401
    embedding_only,  # noqa: F401
)
from ivr_bench.routers.needle import router as needle_router  # noqa: F401
from ivr_bench.routers.registry import available, create, register

__all__ = ["Router", "available", "create", "register"]
