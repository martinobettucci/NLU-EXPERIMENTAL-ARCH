"""Registre des architectures de routage.

Ajouter une architecture consiste a ecrire une classe conforme au protocole
`Router`, l'enregistrer ici, et deposer un fichier dans `config/architectures/`.
Aucun autre fichier du depot n'a besoin d'etre modifie : c'est le critere
d'acceptation 14 de la specification.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from ivr_bench.routers.base import Router

RouterFactory = Callable[..., Router]

_REGISTRY: dict[str, RouterFactory] = {}


def register(name: str) -> Callable[[RouterFactory], RouterFactory]:
    """Enregistre une fabrique de routeur sous un nom stable."""

    def decorator(factory: RouterFactory) -> RouterFactory:
        if name in _REGISTRY:
            raise ValueError(f"architecture deja enregistree : {name}")
        _REGISTRY[name] = factory
        return factory

    return decorator


def available() -> tuple[str, ...]:
    """Noms des architectures enregistrees, triés."""
    return tuple(sorted(_REGISTRY))


def create(name: str, **kwargs: Any) -> Router:
    """Instancie une architecture par son nom.

    Les options de configuration sont filtrees sur la signature de la fabrique :
    une campagne peut passer les reglages du retriever a toutes les
    architectures sans que celles qui n'en ont pas l'usage n'echouent. C'est ce
    qui evite d'enumerer dans l'appelant quelle architecture accepte quoi — une
    liste qui devient fausse des qu'on en ajoute une.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(available()) or "aucune"
        raise KeyError(f"architecture inconnue : {name}. Enregistrees : {known}") from None

    parameters = inspect.signature(factory).parameters
    accepts_everything = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    accepted = (
        kwargs
        if accepts_everything
        else {key: value for key, value in kwargs.items() if key in parameters}
    )
    return factory(**accepted)


def registry() -> Mapping[str, RouterFactory]:
    """Vue en lecture seule du registre."""
    return dict(_REGISTRY)
