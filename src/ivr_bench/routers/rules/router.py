"""A0 — baseline deterministe (§9).

Regles, expressions regulieres et mots cles. Son role n'est pas de gagner : il
est d'etablir un plancher et de verifier que les jeux de donnees ne sont pas
trivialement separables. Une baseline de regles qui atteindrait 95 % signalerait
un corpus trop facile, pas une bonne architecture.

L'ordre d'evaluation est significatif : la securite passe avant le metier, et le
contraste (« je ne veux pas annuler, je veux changer l'heure ») avant les motifs
generiques qu'il contient.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ivr_bench.domain.catalog import default_catalog, load_safety_policy
from ivr_bench.domain.models import RouterPrediction, SessionContext, ToolDefinition
from ivr_bench.generators.practitioners import load_name_banks
from ivr_bench.resolver.phonetics import strip_accents
from ivr_bench.routers.registry import register
from ivr_bench.routers.rules import extraction

# Sujets d'information generale, du plus specifique au plus generique.
_TOPIC_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("opening_hours", ("heure ouvre", "horaires", "ouvert", "fermez", "ouvre")),
    ("parking", ("parking", "me garer", "stationnement")),
    ("accessibility", ("fauteuil", "mobilite reduite", "ascenseur", "accessible")),
    ("access", ("bus", "transport", "gare", "metro", "venir chez vous")),
    ("required_documents", ("papiers", "documents", "apporter", "amener")),
    ("payment", ("payer", "carte bleue", "carte bancaire", "regle", "especes")),
    ("accepted_insurance", ("mutuelle", "tiers payant", "rembours", "carte vitale")),
    ("pricing", ("tarif", "prix", "combien coute")),
    ("teleconsultation", ("teleconsultation", "distance", "visio")),
    ("delays", ("delai", "attente", "combien de temps")),
    ("address", ("adresse", "ou se trouve", "situes", "ou est le cabinet")),
    ("contact", ("joindre", "telephone", "secretariat ouvert")),
)

_LIST_CUES = (
    "mes prochains rendez-vous",
    "prochain rendez-vous",
    "de prevu",
    "j'ai quoi",
    "quand est mon",
    "quand je dois venir",
    "mes rendez-vous",
    "deja un rendez-vous",
    "verifier",
    "rappelez-moi",
    "j'ai oublie",
    "c'est quand",
)

_RESCHEDULE_CUES = (
    "deplac",
    "decal",
    "reporter",
    "modifier",
    "changer l'heure",
    "changer d'heure",
    "basculer",
    "pousser",
    "avancer",
    "ne peux plus venir",
    "a la place",
    "mettez-le",
    "mettez le",
)

_NEW_CUES = (
    "prendre rendez-vous",
    "prendre un rendez-vous",
    "un rendez-vous",
    "une consultation",
    "consulter",
    "voir le docteur",
    "voir la docteure",
    "creneau",
    "prenez-moi",
    "je veux voir",
    "besoin d'un",
    "venir vous voir",
)

_HUMAN_CUES = (
    "parler a quelqu'un",
    "parler a une personne",
    "vraie personne",
    "secretariat",
    "conseiller",
    "un humain",
    "une personne",
)

_OUT_OF_SCOPE_CUES = (
    "temps fera",
    "meteo",
    "banque",
    "histoire",
    "abonnement",
    "restaurant",
    "capitale",
    "pizza",
    "train",
    "chanter",
    "electricite",
    "resiliez",
)


def _contains(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


class RulesRouter:
    """Routeur a base de regles, sans aucun apprentissage."""

    name = "rules"

    def __init__(self) -> None:
        self._catalog = default_catalog()
        self._policy = load_safety_policy()
        self._specialties = list(load_name_banks()["specialties"])

        self._emergency_cues: list[tuple[str, str]] = [
            (strip_accents(cue).lower(), category)
            for category, block in self._policy["emergency"]["categories"].items()
            for cue in block["cues"]
        ]
        self._clinical_cues = [
            strip_accents(cue).lower() for cue in self._policy["clinical_refusal"]["cues"]
        ]

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        started = time.perf_counter()
        allowed = {tool.name for tool in tools} or set(self._catalog.names)
        text = strip_accents(utterance).lower()

        decided, arguments = self._decide(utterance, text)
        name: str | None = decided
        # Les architectures hybrides ne transmettent qu'un sous-ensemble : une
        # fonction hors de ce sous-ensemble ne peut pas etre choisie.
        if name not in allowed:
            name, arguments = ("no_tool", {}) if "no_tool" in allowed else (None, {})

        return RouterPrediction(
            tool_name=name,
            arguments=arguments,
            confidence=None,
            raw_output=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"router": self.name},
        )

    # -- decision -----------------------------------------------------------

    def _decide(self, utterance: str, text: str) -> tuple[str, dict[str, Any]]:
        # 1. Securite d'abord : une urgence ne doit jamais etre traitee comme une
        #    demande administrative, quelle que soit la suite de la phrase.
        for cue, category in self._emergency_cues:
            if cue in text:
                return "emergency_handoff", {"reason_category": category}

        # 2. Demande de nature clinique : le systeme ne repond pas, il transfere.
        if _contains(text, tuple(self._clinical_cues)):
            return "human_handoff", {"reason": "unsupported_request"}

        if _contains(text, _OUT_OF_SCOPE_CUES):
            return "no_tool", {}

        if _contains(text, _HUMAN_CUES):
            return "human_handoff", {"reason": "user_requested_human"}

        # 3. La modification passe avant la prise : « je ne veux pas annuler, je
        #    veux changer l'heure » contient les indices des deux.
        if _contains(text, _RESCHEDULE_CUES):
            return "request_appointment_reschedule", {
                "appointment_reference": None,
                "practitioner_name": extraction.extract_practitioner(utterance),
                "current_date": None,
                "preferred_new_date": extraction.extract_date(utterance),
                "preferred_new_time": extraction.extract_time(utterance),
            }

        # 4. La consultation avant la prise : « est-ce que j'ai deja un
        #    rendez-vous avec... » contient « un rendez-vous ».
        if _contains(text, _LIST_CUES) and not re.search(r"\b(je veux|je voudrais)\s+un\b", text):
            return "list_appointments", {
                "date_from": extraction.extract_date(utterance),
                "date_to": None,
                "practitioner_name": extraction.extract_practitioner(utterance),
            }

        if _contains(text, _NEW_CUES):
            return "request_new_appointment", {
                "practitioner_name": extraction.extract_practitioner(utterance),
                "specialty": extraction.extract_specialty(utterance, self._specialties),
                "preferred_date": extraction.extract_date(utterance),
                "preferred_time": extraction.extract_time(utterance),
                "reason": None,
            }

        for topic, cues in _TOPIC_CUES:
            if _contains(text, cues):
                return "answer_general_information", {"topic": topic}

        # Rien ne correspond : on l'assume plutot que de deviner.
        return "no_tool", {}


register("rules")(RulesRouter)
