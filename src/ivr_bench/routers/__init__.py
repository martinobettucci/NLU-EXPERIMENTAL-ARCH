"""Architectures de routage comparees.

L'import de ce paquet enregistre les architectures disponibles : un routeur
absent du registre serait invisible pour le banc d'essai.
"""

from ivr_bench.routers import rules  # noqa: F401
from ivr_bench.routers.base import Router
from ivr_bench.routers.hybrid import embedding_only  # noqa: F401
from ivr_bench.routers.registry import available, create, register

__all__ = ["Router", "available", "create", "register"]
