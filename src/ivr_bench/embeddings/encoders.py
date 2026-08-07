"""Encodeurs reels du banc d'essai.

Aucun substitut : si les poids manquent, l'encodeur echoue avec un message
explicite. Un encodeur factice produirait des courbes lisibles et fausses, ce
qui est pire que pas de courbe du tout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ivr_bench.embeddings.base import (
    Vectors,
    l2_normalize,
    model_cache_dir,
    require_token,
)

# Dimensions natives de la representation Matryoshka d'EmbeddingGemma. Toute
# autre dimension exige une projection apprise (§10.4).
GEMMA_MRL_DIMENSIONS: tuple[int, ...] = (768, 512, 256, 128)


@dataclass(frozen=True)
class EncoderSpec:
    """Description d'un encodeur, journalisee avec le run."""

    name: str
    model_id: str
    dimension: int
    #: Vrai si la dimension est obtenue par troncature Matryoshka legitime.
    matryoshka: bool
    gated: bool


# Catalogue des encodeurs comparables. Le second sert a verifier que le cout du
# retriever ne depasse pas celui du routeur final (§10.4).
ENCODERS: dict[str, EncoderSpec] = {
    "embeddinggemma_768": EncoderSpec(
        "embeddinggemma_768",
        "google/embeddinggemma-300m",
        768,
        matryoshka=True,
        gated=True,
    ),
    "embeddinggemma_512": EncoderSpec(
        "embeddinggemma_512",
        "google/embeddinggemma-300m",
        512,
        matryoshka=True,
        gated=True,
    ),
    "embeddinggemma_256": EncoderSpec(
        "embeddinggemma_256",
        "google/embeddinggemma-300m",
        256,
        matryoshka=True,
        gated=True,
    ),
    "embeddinggemma_128": EncoderSpec(
        "embeddinggemma_128",
        "google/embeddinggemma-300m",
        128,
        matryoshka=True,
        gated=True,
    ),
    "minilm_multilingual_384": EncoderSpec(
        "minilm_multilingual_384",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        384,
        matryoshka=False,
        gated=False,
    ),
}


class SentenceTransformerEncoder:
    """Encodeur adosse a sentence-transformers, sur CPU."""

    def __init__(self, spec: EncoderSpec) -> None:
        self._spec = spec
        self.name = spec.name
        self.dimension = spec.dimension
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - depend de l'extra installe
            raise RuntimeError(
                "sentence-transformers absent : installez l'extra 'embeddings'."
            ) from error

        kwargs: dict[str, Any] = {
            "cache_folder": str(model_cache_dir()),
            "device": "cpu",
        }
        if self._spec.gated:
            kwargs["token"] = require_token()
        # La troncature Matryoshka est demandee au modele lui-meme : c'est la
        # seule reduction autorisee sans projection apprise.
        if self._spec.matryoshka and self._spec.dimension not in (
            0,
            GEMMA_MRL_DIMENSIONS[0],
        ):
            kwargs["truncate_dim"] = self._spec.dimension

        self._model = SentenceTransformer(self._spec.model_id, **kwargs)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        produced = np.asarray(vectors, dtype=np.float32)
        if produced.shape[1] != self.dimension:
            raise RuntimeError(
                f"{self.name} a produit {produced.shape[1]} dimensions au lieu de "
                f"{self.dimension} : la troncature Matryoshka n'a pas ete appliquee."
            )
        return l2_normalize(produced)


def create_encoder(name: str) -> SentenceTransformerEncoder:
    """Instancie un encodeur du catalogue."""
    try:
        spec = ENCODERS[name]
    except KeyError:
        raise KeyError(
            f"encodeur inconnu : {name}. Disponibles : {', '.join(sorted(ENCODERS))}"
        ) from None
    return SentenceTransformerEncoder(spec)
