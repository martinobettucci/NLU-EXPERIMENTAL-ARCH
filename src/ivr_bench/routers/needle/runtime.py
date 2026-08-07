"""Runtime Needle (26M parametres, encodeur-decodeur JAX).

Le modele est charge une seule fois par processus : le recharger a chaque appel
melangerait le cout de demarrage a la latence d'inference et rendrait toute
mesure de p95 illisible.

Avertissement de mesure : les latences relevees ici viennent du runtime JAX sur
CPU, pas du moteur natif de Cactus, qui n'est pas distribue pour cette
plateforme. Les chiffres publies sont donc un plancher pessimiste et ne sont pas
comparables aux debits annonces par l'auteur du modele.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ivr_bench.domain.models import ToolDefinition
from ivr_bench.domain.paths import weights_dir

NEEDLE_REPO = "Cactus-Compute/needle"


# Fenetre d'encodage par defaut de Needle. Le catalogue complet du domaine pese
# a lui seul plus que cela : le laisser a sa valeur d'origine tronquerait
# silencieusement les dernieres fonctions, et A2 serait juge sur un catalogue
# mutile plutot que sur ses merites.
#
# Verification faite avant d'elargir : sur un appel a deux fonctions (431
# tokens), la sortie est identique a 1024 et a 2048. Elargir la fenetre ne
# change donc pas le comportement du modele ; cela evite seulement la
# troncature. Ce qui degrade A2 est le catalogue complet lui-meme, pas ce
# reglage — et c'est precisement l'effet que la preselection supprime.
DEFAULT_ENCODER_WINDOW = 1024

# Marge pour l'enonce lui-meme, ajoute aux definitions d'outils.
_QUERY_HEADROOM = 256


@dataclass(frozen=True)
class NeedleCall:
    raw: str
    latency_ms: float
    #: Taille reelle de l'entree encodeur, mesuree et publiee : c'est le cout que
    #: la preselection semantique fait justement baisser.
    prompt_tokens: int
    encoder_window: int


def tools_payload(tools: list[ToolDefinition]) -> str:
    """Traduit le catalogue canonique vers le schema attendu par Needle.

    Les definitions ne sont jamais ecrites a la main pour ce modele : elles
    derivent de `functions.yaml`, comme pour toutes les autres architectures.
    C'est ce qui rend la comparaison equitable.
    """
    payload = []
    for tool in tools:
        parameters = {
            parameter.name: {
                "type": parameter.type,
                "description": parameter.description,
                "required": parameter.required,
            }
            for parameter in tool.parameters
        }
        payload.append(
            {"name": tool.name, "description": tool.description, "parameters": parameters}
        )
    return json.dumps(payload, ensure_ascii=False)


def checkpoint_path() -> Path:
    """Emplacement du checkpoint, telecharge explicitement au prealable."""
    cache = Path(os.environ.get("HF_HOME", str(weights_dir() / "huggingface")))
    found = sorted(cache.glob("**/needle.pkl"))
    if not found:
        raise FileNotFoundError("checkpoint Needle absent. Lancez 'ivr-bench models download'.")
    return found[0]


@lru_cache(maxsize=1)
def _loaded() -> tuple[Any, Any, Any]:
    try:
        from needle import SimpleAttentionNetwork, get_tokenizer, load_checkpoint
    except ImportError as error:  # pragma: no cover - depend de l'extra installe
        raise RuntimeError("paquet 'needle' absent : installez l'extra 'needle'.") from error

    params, config = load_checkpoint(str(checkpoint_path()))
    return SimpleAttentionNetwork(config), params, get_tokenizer()


def warmup() -> float:
    """Force la compilation JAX avant toute mesure.

    Sans cela, le premier enonce d'une campagne porterait plusieurs secondes de
    compilation et ecraserait la moyenne de latence.
    """
    started = time.perf_counter()
    call("bonjour", tools_payload([]))
    return (time.perf_counter() - started) * 1000.0


def count_tokens(text: str) -> int:
    """Longueur en tokens, telle que le modele la verra."""
    _, _, tokenizer = _loaded()
    return len(tokenizer.encode(text))


def call(query: str, tools_json: str, max_gen_len: int = 128) -> NeedleCall:
    """Un appel d'outil, sur l'enonce original."""
    from needle import generate

    model, params, tokenizer = _loaded()
    prompt_tokens = len(tokenizer.encode(tools_json)) + len(tokenizer.encode(query))
    # La fenetre s'adapte a ce qu'on transmet reellement. Une troncature
    # silencieuse ferait passer une limite de contexte pour une erreur de
    # jugement du modele.
    window = DEFAULT_ENCODER_WINDOW
    while window < prompt_tokens + _QUERY_HEADROOM:
        window *= 2

    started = time.perf_counter()
    raw = generate(
        model,
        params,
        tokenizer,
        query=query,
        tools=tools_json,
        max_gen_len=max_gen_len,
        max_enc_len=window,
        stream=False,
    )
    return NeedleCall(
        raw=raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False),
        latency_ms=(time.perf_counter() - started) * 1000.0,
        prompt_tokens=prompt_tokens,
        encoder_window=window,
    )
