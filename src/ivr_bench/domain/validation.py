"""Validation des sorties de modeles (§24).

Chaine imposee : parseur strict, normalisation minimale, validation Pydantic,
validation JSON Schema, validation metier.

Le parseur ne corrige jamais une sortie au point de masquer une erreur du modele.
Les reparations sont deterministes, minimales, et surtout **tracees** : une sortie
reparee est comptabilisee a part, dans une colonne distincte des resultats
principaux. C'est la difference entre mesurer un modele et le flatter.
"""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaValidationError

from ivr_bench.domain.models import FunctionCatalog, ValidationOutcome

# Cloture de bloc de code, frequente en sortie de modele generatif.
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)
# Virgule terminale avant une accolade ou un crochet fermant.
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

# Cles acceptees pour designer la fonction appelee, selon les conventions des
# differents modeles. Toute forme autre que 'name' compte comme reparation.
_NAME_KEYS = ("name", "tool_name", "function", "function_name")
_ARGUMENT_KEYS = ("arguments", "parameters", "args")


class OutputValidator:
    """Valide une sortie de routeur contre le catalogue canonique."""

    def __init__(self, catalog: FunctionCatalog) -> None:
        self._catalog = catalog
        self._validators = {
            definition.name: Draft202012Validator(definition.json_schema())
            for definition in catalog.functions
        }
        self._forbidden_arguments = {*catalog.session_injected, "practitioner_id"}

    def validate(self, raw: str | dict[str, Any] | None) -> ValidationOutcome:
        """Applique la chaine complete et renvoie un verdict explicite."""
        repairs: list[str] = []

        payload = self._parse(raw, repairs)
        if payload is None:
            return ValidationOutcome(validity="invalid", errors=("sortie non analysable en JSON",))

        name = self._extract_name(payload, repairs)
        if name is None:
            return ValidationOutcome(
                validity="invalid", errors=("nom de fonction absent de la sortie",)
            )

        if name not in self._catalog.names:
            return ValidationOutcome(
                validity="invalid",
                tool_name=name,
                errors=(f"fonction hors catalogue : {name}",),
            )

        arguments = self._extract_arguments(payload, repairs)
        if arguments is None:
            return ValidationOutcome(
                validity="invalid",
                tool_name=name,
                errors=("bloc d'arguments illisible",),
            )

        errors = [
            *self._business_errors(name, arguments),
            *self._schema_errors(name, arguments),
        ]
        if errors:
            return ValidationOutcome(
                validity="invalid",
                tool_name=name,
                arguments=arguments,
                errors=tuple(errors),
                repairs=tuple(repairs),
            )

        return ValidationOutcome(
            validity="repaired" if repairs else "native",
            tool_name=name,
            arguments=arguments,
            repairs=tuple(repairs),
        )

    def strip_unpersisted(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Retire les arguments qui ne doivent pas etre conserves (motif de soin)."""
        definition = self._catalog.get(tool_name)
        keep = set(definition.persisted_parameters)
        return {key: value for key, value in arguments.items() if key in keep}

    # -- etapes internes ----------------------------------------------------

    def _parse(self, raw: str | dict[str, Any] | None, repairs: list[str]) -> dict[str, Any] | None:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw

        text = raw.strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = self._parse_after_minimal_repair(text, repairs)

        if isinstance(parsed, list):
            # Un appel unique reste attendu : une liste d'un seul element est
            # toleree, une liste plus longue est une reponse differente de la
            # demande et n'a pas a etre silencieusement tronquee.
            if len(parsed) != 1:
                return None
            repairs.append("appel unique extrait d'une liste")
            parsed = parsed[0]

        return parsed if isinstance(parsed, dict) else None

    def _parse_after_minimal_repair(self, text: str, repairs: list[str]) -> Any:
        candidate = text
        fenced = _CODE_FENCE.match(candidate)
        if fenced:
            candidate = fenced.group("body")
            repairs.append("bloc de code retire")

        start, end = candidate.find("{"), candidate.rfind("}")
        surrounded = start > 0 or (end != -1 and end < len(candidate) - 1)
        if surrounded and start != -1 and end != -1 and start < end:
            candidate = candidate[start : end + 1]
            repairs.append("texte hors objet JSON retire")

        repaired = _TRAILING_COMMA.sub(r"\1", candidate)
        if repaired != candidate:
            candidate = repaired
            repairs.append("virgule terminale retiree")

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _extract_name(self, payload: dict[str, Any], repairs: list[str]) -> str | None:
        for index, key in enumerate(_NAME_KEYS):
            value = payload.get(key)
            if isinstance(value, str):
                if index > 0:
                    repairs.append(f"cle de fonction '{key}' normalisee en 'name'")
                return value.strip()
        return None

    def _extract_arguments(
        self, payload: dict[str, Any], repairs: list[str]
    ) -> dict[str, Any] | None:
        for index, key in enumerate(_ARGUMENT_KEYS):
            if key in payload:
                value = payload[key]
                if value is None:
                    repairs.append("bloc d'arguments nul assimile a un objet vide")
                    return {}
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        return None
                    repairs.append("arguments encodes en chaine decodes")
                if not isinstance(value, dict):
                    return None
                if index > 0:
                    repairs.append(f"cle d'arguments '{key}' normalisee en 'arguments'")
                return value

        repairs.append("bloc d'arguments absent assimile a un objet vide")
        return {}

    def _business_errors(self, name: str, arguments: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        definition = self._catalog.get(name)

        if not definition.executable and arguments:
            errors.append(f"{name} ne prend aucun argument")

        # Un modele n'a aucune autorite pour produire une identite : ni le patient,
        # injecte par la session, ni l'identifiant du praticien, qui releve du
        # resolveur (§25.3, §25.4).
        forbidden = sorted(self._forbidden_arguments.intersection(arguments))
        if forbidden:
            errors.append(f"arguments interdits produits par le modele : {forbidden}")

        return errors

    def _schema_errors(self, name: str, arguments: dict[str, Any]) -> list[str]:
        validator = self._validators[name]
        errors: list[str] = []
        for error in sorted(validator.iter_errors(arguments), key=_schema_error_key):
            location = ".".join(str(part) for part in error.absolute_path) or "(racine)"
            errors.append(f"{location} : {error.message}")
        return errors


def _schema_error_key(error: SchemaValidationError) -> str:
    return ".".join(str(part) for part in error.absolute_path)
