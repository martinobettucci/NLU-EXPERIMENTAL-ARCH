"""Recuperation explicite des poids (§33).

Rien ne se telecharge implicitement pendant les tests. Les poids arrivent par
cette commande, dans un cache configure, et leur absence fait echouer une
campagne au lieu de la faire basculer sur un substitut.
"""

from __future__ import annotations

from dataclasses import dataclass

from ivr_bench.embeddings.base import model_cache_dir, require_token
from ivr_bench.embeddings.encoders import ENCODERS


@dataclass(frozen=True)
class DownloadOutcome:
    model_id: str
    status: str
    detail: str = ""


# Modeles necessaires par profil. Le profil court se limite a ce que la campagne
# reduite utilise reellement.
PROFILE_MODELS: dict[str, tuple[str, ...]] = {
    "smoke": ("google/embeddinggemma-300m", "Cactus-Compute/needle"),
    "full": (
        "google/embeddinggemma-300m",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "Cactus-Compute/needle",
        "google/functiongemma-270m-it",
    ),
}

# Fichiers a recuperer par depot : le checkpoint JAX de Needle suffit, ses
# variantes quantifiees alourdiraient le cache sans servir la campagne.
_ALLOW_PATTERNS: dict[str, list[str]] = {
    "Cactus-Compute/needle": ["needle.pkl", "tokenizer*", "*.model", "*.json", "*.vocab"],
    # Les variantes embarquees (LiteRT, ONNX) ne servent pas la campagne CPU.
    "google/functiongemma-270m-it": [
        "*.safetensors",
        "*.json",
        "*.jinja",
        "tokenizer.model",
    ],
}

# Modeles sous licence Gemma : leur telechargement exige un jeton acceptant la
# licence.
_GATED = {spec.model_id for spec in ENCODERS.values() if spec.gated}


def download(profile: str = "full") -> list[DownloadOutcome]:
    """Telecharge les poids du profil demande."""
    try:
        models = PROFILE_MODELS[profile]
    except KeyError:
        raise KeyError(
            f"profil inconnu : {profile}. Disponibles : {', '.join(sorted(PROFILE_MODELS))}"
        ) from None

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - depend de l'extra installe
        raise RuntimeError("huggingface-hub absent : installez l'extra 'embeddings'.") from error

    cache = model_cache_dir()
    outcomes: list[DownloadOutcome] = []
    for model_id in models:
        token = require_token() if model_id in _GATED else None
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache),
            token=token,
            allow_patterns=_ALLOW_PATTERNS.get(model_id),
            # Les poids alternatifs alourdissent le cache sans servir la campagne.
            ignore_patterns=["*.onnx", "*.gguf", "openvino/*", "*.mlmodel"],
        )
        outcomes.append(DownloadOutcome(model_id=model_id, status="ok", detail=path))
    return outcomes
