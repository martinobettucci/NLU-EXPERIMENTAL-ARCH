"""Contrat commun a toutes les architectures comparees (§9).

Un routeur transforme un enonce en appel de fonction atomique. Il ne planifie pas,
n'execute rien, n'appelle pas le backend et ne resout aucune identite. Cette
frontiere est ce qui permet de comparer equitablement DIET, Needle, FunctionGemma
et les architectures hybrides.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ivr_bench.domain.models import RouterPrediction, SessionContext, ToolDefinition


@runtime_checkable
class Router(Protocol):
    """Interface unique des architectures de routage."""

    #: Nom stable, identique a la cle du registre et au fichier de configuration.
    name: str

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        """Selectionne une fonction et extrait ses arguments.

        Le routeur renvoie `tool_name=None` uniquement s'il ne peut rien decider ;
        le rejet explicite d'une demande hors perimetre passe par `no_tool`.
        Il n'invente jamais la valeur d'un argument absent de l'enonce : un
        argument non exprime vaut `None`, et c'est au gestionnaire de dialogue de
        poser la question suivante (§12).
        """
        ...
