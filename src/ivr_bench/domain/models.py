"""Contrats de donnees du banc d'essai.

Ces modeles sont le langage commun de toutes les architectures comparees. Un
routeur produit une `RouterPrediction`, jamais une structure maison : c'est ce qui
rend comparables des systemes aussi differents que DIET, Needle et FunctionGemma.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Statuts renvoyes par le resolveur de praticien (§8).
ResolutionStatus = Literal["resolved", "ambiguous", "not_found", "missing"]

# Classement de la validite d'une sortie de modele (§24). Les resultats principaux
# n'utilisent que 'native' ; 'repaired' apparait dans une colonne distincte.
OutputValidity = Literal["native", "repaired", "invalid"]


class ToolParameter(BaseModel):
    """Un argument de fonction metier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    type: str = "string"
    nullable: bool = True
    description: str = ""
    enum: tuple[str, ...] | None = None
    # Certains champs ne doivent jamais etre conserves dans les resultats, meme
    # correctement extraits : le motif de consultation releve du soin (§5.3).
    persisted: bool = True

    @property
    def required(self) -> bool:
        return not self.nullable

    def json_schema(self) -> dict[str, Any]:
        """Schema JSON de cet argument, la nullabilite etant explicite."""
        types: list[str] = [self.type]
        if self.nullable:
            types.append("null")
        schema: dict[str, Any] = {"type": types if len(types) > 1 else self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum is not None:
            schema["enum"] = [*self.enum, None] if self.nullable else list(self.enum)
        return schema


class ToolDefinition(BaseModel):
    """Une fonction metier du catalogue canonique."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    executable: bool = True
    parameters: tuple[ToolParameter, ...] = ()
    positive_seeds: tuple[str, ...] = ()
    confusable_with: tuple[str, ...] = ()
    unsupported_examples: tuple[str, ...] = ()
    safety_notes: str = ""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def required_parameters(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters if parameter.required)

    @property
    def persisted_parameters(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters if parameter.persisted)

    def parameter(self, name: str) -> ToolParameter | None:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        return None

    def json_schema(self) -> dict[str, Any]:
        """Schema JSON des arguments, utilise par la validation stricte (§24)."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                parameter.name: parameter.json_schema() for parameter in self.parameters
            },
            "required": list(self.required_parameters),
            "additionalProperties": False,
        }


class FunctionCatalog(BaseModel):
    """Catalogue metier complet, charge depuis `config/domain/functions.yaml`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    locale: str = "fr-FR"
    timezone: str = "Europe/Paris"
    # Valeurs injectees par la session, jamais extraites de la parole (§25.3).
    session_injected: tuple[str, ...] = ()
    functions: tuple[ToolDefinition, ...]

    @field_validator("functions")
    @classmethod
    def _names_are_unique(cls, value: tuple[ToolDefinition, ...]) -> tuple[ToolDefinition, ...]:
        names = [definition.name for definition in value]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"fonctions dupliquees dans le catalogue : {sorted(duplicates)}")
        return value

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.functions)

    @property
    def executable_names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.functions if definition.executable)

    def get(self, name: str) -> ToolDefinition:
        for definition in self.functions:
            if definition.name == name:
                return definition
        raise KeyError(f"fonction inconnue : {name}")

    def subset(self, names: list[str] | tuple[str, ...]) -> tuple[ToolDefinition, ...]:
        """Sous-catalogue transmis a un modele, coeur des architectures hybrides."""
        return tuple(self.get(name) for name in names)


class Practitioner(BaseModel):
    """Praticien synthetique du catalogue (§7).

    `split` isole les praticiens reserves au test : ils n'apparaissent dans aucun
    enonce d'entrainement, ce qui verifie que le routeur extrait un nom prononce
    au lieu de memoriser une liste fermee de personnes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    practitioner_id: str
    display_name: str
    first_name: str
    last_name: str
    specialty: str
    site_id: str
    aliases: tuple[str, ...] = ()
    phonetic_aliases: tuple[str, ...] = ()
    split: Literal["train", "test"] = "train"
    #: Categorie de difficulte du nom : accents, apostrophe, compose, homophone...
    name_category: str = "french_common"
    #: Groupe d'homophones auquel ce nom appartient, le cas echeant.
    homophone_group: str | None = None


class ToolCandidate(BaseModel):
    """Une fonction candidate proposee par un retriever, avec son score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    score: float
    rank: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionContext(BaseModel):
    """Etat conversationnel (§13).

    Le contexte est detenu par le gestionnaire de dialogue deterministe : un routeur
    le lit, ne le modifie pas.
    """

    model_config = ConfigDict(extra="forbid")

    active_workflow: str | None = None
    pending_slots: tuple[str, ...] = ()
    confirmed_slots: dict[str, Any] = Field(default_factory=dict)
    candidate_practitioners: tuple[str, ...] = ()
    # Provient de l'authentification, jamais de la transcription.
    patient_id: str | None = None
    locale: str = "fr-FR"
    timezone: str = "Europe/Paris"
    consecutive_failures: int = 0


class RouterPrediction(BaseModel):
    """Sortie canonique d'un routeur (§9)."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str | None
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    candidates: tuple[ToolCandidate, ...] = ()
    raw_output: str | dict[str, Any] | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PractitionerCandidate(BaseModel):
    """Praticien propose par le resolveur."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    practitioner_id: str
    display_name: str
    score: float


class PractitionerResolution(BaseModel):
    """Resultat de la resolution du praticien (§8).

    Le routeur n'a jamais acces a ce module : il transmet un nom prononce, le
    resolveur seul decide de l'identite.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ResolutionStatus
    practitioner_id: str | None = None
    confidence: float | None = None
    candidates: tuple[PractitionerCandidate, ...] = ()


class ValidationOutcome(BaseModel):
    """Verdict de la chaine de validation d'une sortie de modele (§24)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    validity: OutputValidity
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    errors: tuple[str, ...] = ()
    repairs: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return self.validity != "invalid"
