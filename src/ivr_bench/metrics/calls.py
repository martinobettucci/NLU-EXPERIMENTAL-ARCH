"""Taux d'appel exact : la fonction *et* tous ses arguments (§17.2).

Pourquoi une metrique de plus. L'exactitude d'arguments se compte cle par cle
et seulement sur les cas ou la fonction est correcte : elle recompense donc la
justesse partielle, et elle se calcule sur un sous-ensemble different pour
chaque architecture. Un serveur vocal, lui, n'execute pas un demi-appel. Ce que
le systeme rend utilisable, c'est la proportion d'enonces pour lesquels l'appel
complet est juste — et cette proportion se compte sur tout le corpus, donc elle
se compare directement d'une architecture a l'autre.

Les deux metriques ne classent pas les architectures dans le meme ordre, et
c'est precisement pourquoi les deux sont publiees.

La valeur se recalcule depuis `predictions.jsonl`, jamais depuis un resume :
elle existe donc aussi pour les runs anterieurs a son introduction, sans
qu'aucun run ait a etre reecrit (§29.3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalise(value: Any) -> str | None:
    """Meme normalisation que la comparaison d'arguments : casse et espaces."""
    if value is None:
        return None
    text = " ".join(str(value).lower().split())
    return text or None


def is_exact_call(expected: dict[str, Any], predicted: dict[str, Any]) -> bool:
    """L'appel produit est-il celui attendu, arguments compris ?"""
    if predicted.get("tool_name") != expected.get("tool_name"):
        return False
    wanted = expected.get("arguments") or {}
    produced = predicted.get("arguments") or {}
    return all(
        _normalise(wanted.get(key)) == _normalise(produced.get(key))
        for key in set(wanted) | set(produced)
    )


def exact_call_rate(directory: Path) -> float | None:
    """Proportion d'appels entierement corrects dans un run."""
    path = directory / "predictions.jsonl"
    if not path.is_file():
        return None

    total = 0
    exact = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        exact += int(is_exact_call(row.get("expected", {}), row.get("predicted", {})))
    return exact / total if total else None
