"""Resolution du praticien (§8).

Composant strictement separe du routeur. Le routeur extrait ce qui a ete
prononce ; lui seul decide de l'identite, et il a le droit de repondre « je ne
sais pas ». Une resolution unique erronee avec forte confiance est plus grave
qu'une demande de clarification (§17.3) : les seuils sont donc regles pour
clarifier plutot que pour trancher.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rapidfuzz import fuzz, process

from ivr_bench.domain.models import Practitioner, PractitionerCandidate, PractitionerResolution
from ivr_bench.resolver.normalization import comparable
from ivr_bench.resolver.phonetics import phonetic_key

# Ponderation entre ressemblance ecrite et ressemblance sonore. Le lexical domine
# parce qu'il discrimine mieux ; le phonetique rattrape les transcriptions que la
# reconnaissance vocale a deformees sans changer le son.
LEXICAL_WEIGHT = 0.70
PHONETIC_WEIGHT = 0.30

# Nombre de formes lexicales rapprochees examinees avant agregation par praticien.
_LEXICAL_SHORTLIST = 60


class PractitionerResolver:
    """Resout un nom prononce vers un praticien du catalogue."""

    def __init__(
        self,
        practitioners: list[Practitioner],
        clarification_threshold: float = 0.80,
        margin: float = 0.05,
        minimum_score: float = 0.55,
    ) -> None:
        if not practitioners:
            raise ValueError("le resolveur exige un catalogue non vide")
        self._practitioners = practitioners
        self._clarification_threshold = clarification_threshold
        self._margin = margin
        self._minimum_score = minimum_score

        # Index plat des formes prononcables, pour une comparaison vectorisee.
        self._alias_forms: list[str] = []
        self._alias_owner: list[int] = []
        self._phonetic_index: dict[str, list[int]] = defaultdict(list)

        for position, practitioner in enumerate(practitioners):
            forms = {
                comparable(alias) for alias in (*practitioner.aliases, practitioner.display_name)
            }
            forms.discard("")
            for form in sorted(forms):
                self._alias_forms.append(form)
                self._alias_owner.append(position)
            for key in practitioner.phonetic_aliases:
                self._phonetic_index[key].append(position)

    def resolve(
        self,
        spoken_name: str | None,
        specialty: str | None = None,
        site_id: str | None = None,
        top_k: int = 3,
    ) -> PractitionerResolution:
        """Renvoie un statut explicite plutot qu'un identifiant optimiste."""
        query = comparable(spoken_name or "")
        if not query:
            return PractitionerResolution(status="missing")

        scores, phonetic_hits = self._score(query, specialty=specialty, site_id=site_id)
        if not scores:
            return PractitionerResolution(status="not_found")

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        candidates = tuple(
            PractitionerCandidate(
                practitioner_id=self._practitioners[position].practitioner_id,
                display_name=self._practitioners[position].display_name,
                score=round(score, 4),
            )
            for position, score in ranked[:top_k]
        )

        best = ranked[0][1]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        if best < self._minimum_score:
            return PractitionerResolution(status="not_found", candidates=candidates)

        # Si ce qui a ete prononce correspond a plusieurs praticiens, l'ecart
        # lexical entre eux ne prouve rien : il ne reflete que l'orthographe
        # choisie par la reconnaissance vocale, pas ce que l'appelant a dit.
        # `Ray`, `Rey` et `Reï` arrivent tous sous une seule graphie. On ne
        # tranche donc pas : on demande. La specialite ou le site, eux, restent
        # des criteres legitimes et ont deja reduit l'ensemble en amont.
        if len(phonetic_hits) > 1:
            return PractitionerResolution(
                status="ambiguous", confidence=round(best, 4), candidates=candidates
            )

        # Deux conditions, pas une : un score eleve partage par deux homonymes
        # reste une ambiguite, et l'appelant doit trancher lui-meme.
        if best >= self._clarification_threshold and (best - runner_up) >= self._margin:
            return PractitionerResolution(
                status="resolved",
                practitioner_id=candidates[0].practitioner_id,
                confidence=round(best, 4),
                candidates=candidates,
            )

        return PractitionerResolution(
            status="ambiguous", confidence=round(best, 4), candidates=candidates
        )

    # -- interne ------------------------------------------------------------

    def _score(
        self, query: str, specialty: str | None, site_id: str | None
    ) -> tuple[dict[int, float], set[int]]:
        allowed = self._restrict(specialty=specialty, site_id=site_id)

        lexical: dict[int, float] = {}
        for form, score, index in process.extract(
            query,
            self._alias_forms,
            scorer=fuzz.WRatio,
            limit=_LEXICAL_SHORTLIST,
        ):
            del form
            position = self._alias_owner[index]
            if allowed is not None and position not in allowed:
                continue
            ratio = float(score) / 100.0
            if ratio > lexical.get(position, 0.0):
                lexical[position] = ratio

        key = phonetic_key(query)
        phonetic_hits = {
            position
            for position in self._phonetic_index.get(key, [])
            if allowed is None or position in allowed
        }

        combined: dict[int, float] = {}
        for position in set(lexical) | phonetic_hits:
            lexical_score = lexical.get(position, 0.0)
            phonetic_score = 1.0 if position in phonetic_hits else 0.0
            combined[position] = LEXICAL_WEIGHT * lexical_score + PHONETIC_WEIGHT * phonetic_score
        return combined, phonetic_hits

    def _restrict(self, specialty: str | None, site_id: str | None) -> set[int] | None:
        """Filtre par specialite ou site, uniquement s'ils sont exploitables.

        Un filtre qui ne laisserait aucun candidat serait pire que pas de filtre :
        on l'ignore alors, plutot que de repondre « introuvable » a cause d'une
        specialite mal comprise.
        """
        if specialty is None and site_id is None:
            return None

        wanted_specialty = comparable(specialty or "")
        allowed = {
            position
            for position, practitioner in enumerate(self._practitioners)
            if (not wanted_specialty or comparable(practitioner.specialty) == wanted_specialty)
            and (site_id is None or practitioner.site_id == site_id)
        }
        return allowed or None

    def describe(self) -> dict[str, Any]:
        """Parametres effectifs, journalises avec chaque run."""
        return {
            "practitioners": len(self._practitioners),
            "alias_forms": len(self._alias_forms),
            "clarification_threshold": self._clarification_threshold,
            "margin": self._margin,
            "minimum_score": self._minimum_score,
            "lexical_weight": LEXICAL_WEIGHT,
            "phonetic_weight": PHONETIC_WEIGHT,
        }
