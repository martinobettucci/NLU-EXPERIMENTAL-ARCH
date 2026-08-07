"""Encodeurs semantiques interchangeables (§10.4).

Une regle domine ce module : **on ne tronque pas un embedding qui n'a pas ete
entraine pour l'etre**. Reduire la dimension n'est legitime que si le modele a
appris une representation emboitee (Matryoshka), ou si une projection a ete
apprise sur l'entrainement seul. Une troncature arbitraire donnerait des courbes
de dimension qui mesureraient surtout notre imprudence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ivr_bench.domain.paths import weights_dir

Vectors = NDArray[np.float32]


class EmbeddingEncoder(Protocol):
    """Interface commune a tous les encodeurs du banc d'essai."""

    #: Nom stable, journalise avec chaque run.
    name: str
    #: Dimension effective des vecteurs produits.
    dimension: int

    def encode(self, texts: list[str], batch_size: int = 32) -> Vectors:
        """Encode des enonces en vecteurs normalises L2."""
        ...


def l2_normalize(vectors: Vectors) -> Vectors:
    """Normalisation L2, appliquee une seule fois (§10.5).

    Le runtime ne recalcule jamais les embeddings hors ligne : les vecteurs sont
    normalises a la construction, la similarite cosinus devient un simple produit
    scalaire.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Un vecteur nul reste nul plutot que de produire des NaN silencieux.
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


def model_cache_dir() -> Path:
    directory = Path(os.environ.get("HF_HOME", weights_dir() / "huggingface"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def require_token() -> str:
    """Jeton Hugging Face, exige explicitement plutot que devine.

    EmbeddingGemma et FunctionGemma sont sous licence Gemma : sans jeton, la
    suite echoue avec un message clair au lieu de basculer sur un substitut.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN absent. Les modeles Gemma sont sous licence : exportez un "
            "jeton Hugging Face ayant accepte la licence, puis relancez "
            "'ivr-bench models download'."
        )
    return token
