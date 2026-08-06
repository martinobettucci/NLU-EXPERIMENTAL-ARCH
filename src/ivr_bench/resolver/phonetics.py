"""Cle phonetique adaptee au francais.

Les algorithmes classiques (Soundex, Metaphone) sont calibres sur l'anglais : ils
separent `Rey` de `Ray`, ecrasent `ch` et `sch` differemment, et ignorent les
consonnes finales muettes qui sont pourtant la premiere source de confusion en
francais. On implemente donc une transcription approchee, deterministe et
documentee, plutot que d'importer une dependance qui se tromperait ailleurs.

L'objectif n'est pas la justesse phonologique : c'est que deux noms qu'un
appelant prononce de la meme facon produisent la meme cle, pour que le resolveur
demande une clarification au lieu de trancher au hasard.
"""

from __future__ import annotations

import re
import unicodedata

# Regles appliquees dans l'ordre. L'ordre compte : 'ault' avant 'au', 'sch' avant
# 'ch', 'x' avant que 'ch' ne produise lui-meme un 'x'.
_RULES: tuple[tuple[str, str], ...] = (
    # Le 'x' latin devient 'ks' avant que 'ch' n'occupe le symbole 'x'.
    ("x", "ks"),
    # Terminaisons a consonnes muettes : Renault, Renaud et Renaut se rejoignent.
    ("ault", "o"),
    ("auld", "o"),
    ("eaux", "o"),
    ("eau", "o"),
    ("aux", "o"),
    ("au", "o"),
    ("oeu", "e"),
    ("oe", "e"),
    ("ou", "u"),
    # Voyelles nasales.
    ("ain", "in"),
    ("ein", "in"),
    ("aim", "in"),
    ("aon", "an"),
    # Diphtongues rendues par un meme son : Rey, Ray et Rei.
    ("ai", "e"),
    ("ei", "e"),
    ("ay", "e"),
    ("ey", "e"),
    # Consonnes composees.
    ("sch", "x"),
    ("ph", "f"),
    ("ch", "x"),
    ("gn", "n"),
    ("ill", "y"),
    ("qu", "k"),
    ("ck", "k"),
    ("q", "k"),
    ("tion", "sion"),
    # 'c' et 'g' dependent de la voyelle qui suit.
    ("ce", "se"),
    ("ci", "si"),
    ("cy", "si"),
    ("c", "k"),
    ("ge", "je"),
    ("gi", "ji"),
    ("gy", "ji"),
    # Divers.
    ("w", "v"),
    ("y", "i"),
    ("h", ""),
)

# Consonnes finales le plus souvent muettes en francais.
_SILENT_FINALS = ("s", "t", "d", "x", "z", "p")

# Longueur minimale avant de laisser tomber un 'e' final : sans ce garde-fou,
# 'rey' se reduirait a 'r' et perdrait toute valeur discriminante.
_MIN_LENGTH_FOR_MUTE_E = 4

_NON_LETTERS = re.compile(r"[^a-z]+")


def strip_accents(text: str) -> str:
    """Retire les diacritiques sans supprimer les lettres qui les portent."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _collapse_repeats(text: str) -> str:
    collapsed: list[str] = []
    for char in text:
        if not collapsed or collapsed[-1] != char:
            collapsed.append(char)
    return "".join(collapsed)


def phonetic_key(name: str) -> str:
    """Cle phonetique approchee d'un nom.

    Les espaces, apostrophes et traits d'union disparaissent : `Ben Said`,
    `Ben-Said` et `Bensaid` designent la meme personne a l'oreille.
    """
    text = _NON_LETTERS.sub("", strip_accents(name).lower())
    if not text:
        return ""

    for pattern, replacement in _RULES:
        text = text.replace(pattern, replacement)

    # Les doublons se collapsent avant la chute des finales : 'Petitt' doit
    # rejoindre 'Petit', pas s'arreter a 'petit'.
    text = _collapse_repeats(text)

    # Une seule troncature finale, jamais une cascade : 'Bensaid' doit donner
    # 'bense' et non 'ben', sous peine de confondre des noms sans rapport.
    if (len(text) >= 3 and text[-1] in _SILENT_FINALS) or (
        len(text) >= _MIN_LENGTH_FOR_MUTE_E and text.endswith("e")
    ):
        text = text[:-1]

    return text


def phonetic_keys(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Cles distinctes et non vides d'une liste de formes."""
    seen: dict[str, None] = {}
    for name in names:
        key = phonetic_key(name)
        if key:
            seen.setdefault(key, None)
    return tuple(seen)
