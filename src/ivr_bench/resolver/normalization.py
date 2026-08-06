"""Normalisation des noms prononces.

Le routeur transmet ce que l'appelant a dit : « docteur ben said », « madame le
docteur Bensaid », « Dr Bensaid ». Le titre fait partie de l'argument extrait —
on ne le retire donc pas du corpus, seulement au moment de comparer des noms.
"""

from __future__ import annotations

import re

from ivr_bench.resolver.phonetics import strip_accents

# Titres pouvant preceder un nom, du plus long au plus court : « madame le
# docteur » doit disparaitre entierement avant que « docteur » ne s'applique.
_TITLES: tuple[str, ...] = (
    "madame le docteur",
    "monsieur le docteur",
    "madame la docteure",
    "monsieur la docteure",
    "madame le professeur",
    "monsieur le professeur",
    "le docteur",
    "la docteure",
    "le professeur",
    "la professeure",
    "docteure",
    "docteur",
    "professeure",
    "professeur",
    "madame",
    "monsieur",
    "mademoiselle",
    "docteurs",
    "mme",
    "mlle",
    "mr",
    "dr",
    "pr",
)

_PUNCTUATION = re.compile(r"[.,;:!?()\"]+")
_SEPARATORS = re.compile(r"[-'’ʼ]+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Forme comparable d'un nom : sans accent, sans ponctuation, en minuscules.

    Les apostrophes et traits d'union deviennent des espaces plutot que de
    disparaitre : `D'Angelo` doit rester deux jetons, sinon la comparaison
    lexicale confond `Dangelo` avec des noms sans rapport.
    """
    lowered = strip_accents(text).lower()
    without_punctuation = _PUNCTUATION.sub(" ", lowered)
    separated = _SEPARATORS.sub(" ", without_punctuation)
    return _WHITESPACE.sub(" ", separated).strip()


def strip_titles(text: str) -> str:
    """Retire les titres en tete de chaine, de facon repetee et controlee."""
    normalized = normalize(text)
    changed = True
    while changed:
        changed = False
        for title in _TITLES:
            if normalized == title:
                return ""
            if normalized.startswith(f"{title} "):
                normalized = normalized[len(title) + 1 :]
                changed = True
                break
    return normalized.strip()


def comparable(text: str) -> str:
    """Normalisation complete utilisee par le resolveur."""
    return strip_titles(text)
