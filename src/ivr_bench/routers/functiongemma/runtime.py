"""Runtime FunctionGemma 270m-it, sur CPU.

Le format de controle du modele est particulier :

    <start_function_call>call:nom{arg:<escape>valeur<escape>}<end_function_call>

Il n'est pas reconstruit a la main : les definitions d'outils passent par le
gabarit de conversation officiel embarque avec les poids, qui accepte des
schemas JSON standards. Un format maison serait la premiere source d'injustice
envers ce modele.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ivr_bench.domain.models import ToolDefinition

MODEL_ID = "google/functiongemma-270m-it"

_CALL = re.compile(
    r"<start_function_call>\s*call:(?P<name>[\w.]+)\s*\{(?P<body>.*?)\}\s*<end_function_call>",
    re.DOTALL,
)
# Les valeurs sont encadrees par des marqueurs d'echappement litteraux.
_ARGUMENT = re.compile(r"(?P<key>[\w.]+)\s*:\s*<escape>(?P<value>.*?)<escape>", re.DOTALL)


@dataclass(frozen=True)
class GemmaCall:
    raw: str
    parsed: dict[str, Any] | None
    latency_ms: float
    prompt_tokens: int


def tools_payload(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Definitions au format schema JSON, derivees du catalogue canonique."""
    payload: list[dict[str, Any]] = []
    for tool in tools:
        properties: dict[str, Any] = {}
        for parameter in tool.parameters:
            entry: dict[str, Any] = {
                "type": parameter.type,
                "description": parameter.description,
            }
            if parameter.enum:
                entry["enum"] = list(parameter.enum)
            properties[parameter.name] = entry
        # Le gabarit officiel attend l'enveloppe { "type": "function",
        # "function": {...} } : c'est la forme que transformers transmet aux
        # modeles d'appel d'outils.
        payload.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(tool.required_parameters),
                    },
                },
            }
        )
    return payload


def parse_control_format(text: str) -> dict[str, Any] | None:
    """Traduit la sortie de controle vers la structure canonique.

    Renvoie `None` si aucun appel n'est reconnaissable : c'est une sortie
    invalide, et elle doit etre comptee comme telle plutot que rattrapee.
    """
    match = _CALL.search(text)
    if not match:
        return None
    arguments = {
        found.group("key"): found.group("value").strip()
        for found in _ARGUMENT.finditer(match.group("body"))
    }
    return {"name": match.group("name").strip(), "arguments": arguments}


@lru_cache(maxsize=1)
def _loaded() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - depend de l'extra installe
        raise RuntimeError("transformers absent : installez l'extra 'functiongemma'.") from error

    from ivr_bench.embeddings.base import model_cache_dir, require_token

    token = require_token()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=str(model_cache_dir()), token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(model_cache_dir()),
        token=token,
        dtype=torch.float32,
    )
    # transformers n'annote pas eval() ; le mode inference reste indispensable.
    model.eval()  # type: ignore[no-untyped-call]
    return model, tokenizer


def call(query: str, tools: list[dict[str, Any]], max_new_tokens: int = 96) -> GemmaCall:
    """Un appel d'outil sur l'enonce original."""
    import torch

    model, tokenizer = _loaded()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tools=tools or None,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Decodage deterministe : une campagne rejouee doit donner les memes
            # sorties, sinon les intervalles de confiance ne mesurent plus que
            # l'echantillonnage.
            do_sample=False,
        )
    prompt_length = int(inputs["input_ids"].shape[1])
    text = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=False)
    return GemmaCall(
        raw=text,
        parsed=parse_control_format(text),
        latency_ms=(time.perf_counter() - started) * 1000.0,
        prompt_tokens=prompt_length,
    )
