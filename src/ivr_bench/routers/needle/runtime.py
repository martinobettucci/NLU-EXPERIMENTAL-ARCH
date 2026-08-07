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


@dataclass(frozen=True)
class NeedleCall:
    raw: str
    latency_ms: float


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


def call(query: str, tools_json: str, max_gen_len: int = 128) -> NeedleCall:
    """Un appel d'outil, sur l'enonce original."""
    from needle import generate

    model, params, tokenizer = _loaded()
    started = time.perf_counter()
    raw = generate(
        model,
        params,
        tokenizer,
        query=query,
        tools=tools_json,
        max_gen_len=max_gen_len,
        stream=False,
    )
    return NeedleCall(
        raw=raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )
