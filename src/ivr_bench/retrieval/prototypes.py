"""Construction des prototypes par fonction (§10.6).

L'hypothese a tester est explicite : un centroide unique ecrase les modes
semantiques minoritaires. « Je voudrais un rendez-vous » et « prenez-moi le
docteur Rey mardi » appellent la meme fonction par des chemins tres differents ;
leur moyenne ne ressemble a aucun des deux. Plusieurs prototypes gardent ces
modes distincts — encore faut-il le mesurer plutot que l'affirmer.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ivr_bench.embeddings.base import Vectors, l2_normalize

PrototypeStrategy = Literal["all", "centroid", "kmeans", "medoids"]


def centroid(vectors: Vectors) -> Vectors:
    """Prototype unique : la moyenne renormalisee."""
    if vectors.shape[0] == 0:
        raise ValueError("aucun vecteur a resumer")
    return l2_normalize(vectors.mean(axis=0, keepdims=True))


def kmeans(vectors: Vectors, count: int, seed: int = 42, iterations: int = 50) -> Vectors:
    """k-moyennes spherique, deterministe a graine fixee.

    Les vecteurs etant normalises, maximiser le cosinus revient a minimiser la
    distance euclidienne : on reste sur une implementation directe plutot que
    d'ajouter une dependance pour trois boucles.
    """
    if vectors.shape[0] == 0:
        raise ValueError("aucun vecteur a regrouper")
    if count >= vectors.shape[0]:
        return l2_normalize(vectors.copy())

    rng = np.random.default_rng(seed)
    # Initialisation k-means++ : deux centres initiaux voisins produiraient des
    # groupes vides et des prototypes fantomes.
    centres = [vectors[rng.integers(vectors.shape[0])]]
    for _ in range(1, count):
        similarity = np.max(vectors @ np.asarray(centres).T, axis=1)
        distance = np.clip(1.0 - similarity, 0.0, None) ** 2
        total = distance.sum()
        if total <= 0.0:
            centres.append(vectors[rng.integers(vectors.shape[0])])
            continue
        centres.append(vectors[rng.choice(vectors.shape[0], p=distance / total)])

    current = np.asarray(centres, dtype=np.float32)
    for _ in range(iterations):
        assignment = np.argmax(vectors @ current.T, axis=1)
        updated = np.zeros_like(current)
        for index in range(current.shape[0]):
            members = vectors[assignment == index]
            # Un groupe vide garde son centre : le remplacer au hasard rendrait
            # le resultat dependant de l'ordre des donnees.
            updated[index] = members.mean(axis=0) if members.shape[0] else current[index]
        updated = l2_normalize(updated)
        if np.allclose(updated, current, atol=1e-6):
            current = updated
            break
        current = updated
    return current


def medoids(vectors: Vectors, count: int, seed: int = 42) -> Vectors:
    """Prototypes choisis parmi les vecteurs reels, pas construits.

    Un medoid est un enonce existant : contrairement a un centroide, il ne peut
    pas tomber dans une zone de l'espace ou aucune formulation ne vit.
    """
    if vectors.shape[0] == 0:
        raise ValueError("aucun vecteur a resumer")
    if count >= vectors.shape[0]:
        return vectors.copy()

    centres = kmeans(vectors, count, seed=seed)
    assignment = np.argmax(vectors @ centres.T, axis=1)

    chosen: list[Vectors] = []
    for index in range(centres.shape[0]):
        members = np.flatnonzero(assignment == index)
        if members.size == 0:
            continue
        similarity = vectors[members] @ centres[index]
        chosen.append(vectors[members[int(np.argmax(similarity))]])
    return np.asarray(chosen, dtype=np.float32)


def build(
    vectors: Vectors,
    strategy: PrototypeStrategy,
    count: int = 8,
    seed: int = 42,
) -> Vectors:
    """Applique la strategie demandee."""
    if strategy == "all":
        return vectors.copy()
    if strategy == "centroid":
        return centroid(vectors)
    if strategy == "kmeans":
        return kmeans(vectors, count, seed=seed)
    if strategy == "medoids":
        return medoids(vectors, count, seed=seed)
    raise ValueError(f"strategie de prototypes inconnue : {strategy}")
