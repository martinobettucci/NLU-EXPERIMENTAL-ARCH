"""Chargement du catalogue metier canonique et des politiques associees.

Tout le depot lit le domaine par ici. Un adaptateur qui redefinirait les fonctions
dans son coin briserait l'equite de la comparaison : c'est precisement ce que cette
source unique interdit.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ivr_bench.domain.models import FunctionCatalog, ToolDefinition, ToolParameter
from ivr_bench.domain.paths import config_dir


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"fichier de configuration absent : {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"contenu YAML invalide dans {path} : objet attendu")
    return loaded


def _build_parameters(raw: dict[str, Any] | None) -> tuple[ToolParameter, ...]:
    if not raw:
        return ()
    parameters: list[ToolParameter] = []
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"definition d'argument invalide pour '{name}'")
        enum_values = spec.get("enum")
        parameters.append(
            ToolParameter(
                name=name,
                type=spec.get("type", "string"),
                nullable=bool(spec.get("nullable", True)),
                description=str(spec.get("description", "")).strip(),
                enum=tuple(enum_values) if enum_values else None,
                persisted=bool(spec.get("persisted", True)),
            )
        )
    return tuple(parameters)


def _build_definition(raw: dict[str, Any]) -> ToolDefinition:
    return ToolDefinition(
        name=raw["name"],
        description=str(raw.get("description", "")).strip(),
        executable=bool(raw.get("executable", True)),
        parameters=_build_parameters(raw.get("parameters")),
        positive_seeds=tuple(raw.get("positive_seeds") or ()),
        confusable_with=tuple(raw.get("confusable_with") or ()),
        unsupported_examples=tuple(raw.get("unsupported_examples") or ()),
        safety_notes=str(raw.get("safety_notes", "")).strip(),
    )


def load_catalog(path: Path | None = None) -> FunctionCatalog:
    """Charge le catalogue metier canonique."""
    source = path or (config_dir() / "domain" / "functions.yaml")
    raw = _read_yaml(source)

    functions = tuple(_build_definition(entry) for entry in raw.get("functions", []))
    if not functions:
        raise ValueError(f"aucune fonction definie dans {source}")

    catalog = FunctionCatalog(
        version=int(raw.get("version", 1)),
        locale=str(raw.get("locale", "fr-FR")),
        timezone=str(raw.get("timezone", "Europe/Paris")),
        session_injected=tuple(raw.get("session_injected") or ()),
        functions=functions,
    )

    # Une reference croisee erronee passerait inapercue jusqu'au jour ou le
    # generateur produirait des contrastes vers une fonction inexistante.
    known = set(catalog.names)
    for definition in catalog.functions:
        unknown = sorted(set(definition.confusable_with) - known)
        if unknown:
            raise ValueError(
                f"'{definition.name}' declare confusable_with vers des fonctions "
                f"inconnues : {unknown}"
            )
    return catalog


@lru_cache(maxsize=1)
def default_catalog() -> FunctionCatalog:
    """Catalogue par defaut, mis en cache pour le temps du processus."""
    return load_catalog()


def load_safety_policy(path: Path | None = None) -> dict[str, Any]:
    """Charge la politique de securite, appliquee hors modele (§25)."""
    return _read_yaml(path or (config_dir() / "domain" / "safety_policy.yaml"))


def load_general_information(path: Path | None = None) -> dict[str, Any]:
    """Charge le contenu administratif synthetique."""
    return _read_yaml(path or (config_dir() / "domain" / "general_information.yaml"))
