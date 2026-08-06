"""Deformations deterministes appliquees aux enonces.

Un banc d'essai qui ne mesure que des phrases propres ne dit rien d'un serveur
vocal : les appelants hesitent, se reprennent, et la reconnaissance vocale abime
surtout les noms propres. Chaque deformation porte le nom de la sous-suite
qu'elle alimente (§19), pour que la degradation soit imputable a une cause
identifiee et non noyee dans une moyenne.
"""

from __future__ import annotations

import random
import re

from ivr_bench.resolver.phonetics import strip_accents

# Marqueurs d'hesitation courants a l'oral.
_FILLERS = ("euh", "hum", "alors", "en fait", "ben")

# Substitutions typiques d'une reconnaissance vocale francaise : accents perdus,
# liaisons recollees, homophones grammaticaux.
_ASR_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("docteur", "docteure"),
    ("rendez-vous", "rendez vous"),
    ("s'il vous plait", "sil vous plait"),
    ("est-ce que", "esque"),
    ("qu'est-ce que", "kesque"),
    ("c'est", "sait"),
    ("et", "est"),
    ("ça", "sa"),
    ("où", "ou"),
    ("à", "a"),
)


def add_disfluency(text: str, rng: random.Random) -> str:
    """Insere une hesitation ou une reprise, comme a l'oral."""
    words = text.split()
    if len(words) < 3:
        return f"{rng.choice(_FILLERS)} {text}"

    position = rng.randrange(1, len(words))
    mode = rng.randrange(3)
    if mode == 0:
        words.insert(position, rng.choice(_FILLERS))
    elif mode == 1:
        # Repetition d'un mot, tres frequente en parole spontanee.
        words.insert(position, words[position])
    else:
        words.insert(0, rng.choice(_FILLERS) + ",")
    return " ".join(words)


def add_typo(text: str, rng: random.Random) -> str:
    """Introduit une faute de frappe ou d'accord dans un mot suffisamment long."""
    words = text.split()
    candidates = [index for index, word in enumerate(words) if len(word) > 4]
    if not candidates:
        return text

    index = rng.choice(candidates)
    word = words[index]
    position = rng.randrange(1, len(word) - 1)
    mode = rng.randrange(3)
    if mode == 0:
        # Inversion de deux lettres.
        word = word[:position] + word[position + 1] + word[position] + word[position + 2 :]
    elif mode == 1:
        # Lettre manquante.
        word = word[:position] + word[position + 1 :]
    else:
        # Lettre doublee.
        word = word[:position] + word[position] + word[position:]
    words[index] = word
    return " ".join(words)


def add_asr_error(text: str, rng: random.Random) -> str:
    """Simule une transcription automatique imparfaite.

    Les erreurs portent d'abord sur ce que la reconnaissance vocale rate
    reellement : les accents et les noms propres. Le sens reste reconnaissable
    par un humain, ce qui est exactement le cas limite a mesurer.
    """
    result = text
    applicable = [pair for pair in _ASR_SUBSTITUTIONS if pair[0] in result.lower()]
    if applicable:
        source, target = rng.choice(applicable)
        result = re.sub(re.escape(source), target, result, count=1, flags=re.IGNORECASE)

    if rng.random() < 0.5:
        # Perte des accents, symptome classique d'une transcription degradee.
        result = strip_accents(result)

    if result != text:
        return result

    # Aucune regle n'a mordu : plutot que de renvoyer l'enonce intact — qui
    # serait rejete comme doublon et ferait disparaitre la sous-suite ASR des
    # resultats — on applique les deformations que la reconnaissance vocale
    # produit systematiquement : elision des traits d'union et des apostrophes,
    # puis recollage des mots courts.
    result = strip_accents(text).replace("-", " ").replace("'", " ")
    if result == text:
        words = result.split()
        if len(words) > 2:
            position = rng.randrange(len(words) - 1)
            words[position : position + 2] = [words[position] + words[position + 1]]
            result = " ".join(words)
    return result


PERTURBATIONS: dict[str, tuple[str, ...]] = {
    # Chaque deformation alimente la sous-suite du meme nom.
    "spoken_disfluencies": ("disfluency",),
    "typos": ("typo",),
    "asr_clean": ("asr",),
}


def apply(kind: str, text: str, rng: random.Random) -> str:
    if kind == "disfluency":
        return add_disfluency(text, rng)
    if kind == "typo":
        return add_typo(text, rng)
    if kind == "asr":
        return add_asr_error(text, rng)
    raise ValueError(f"deformation inconnue : {kind}")
