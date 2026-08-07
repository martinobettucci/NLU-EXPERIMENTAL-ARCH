"""Conversion du corpus canonique vers le format NLU de Rasa.

DIET travaille sur des intentions et des entites ; le banc d'essai raisonne en
fonctions et arguments. La conversion est deterministe et part exactement du
meme corpus que les autres architectures — c'est la condition d'equite du §A1.

Une intention par fonction, une entite par argument. Les valeurs d'arguments
sont annotees a leur position reelle dans la phrase, si bien que DIET recoit la
meme information que celle qu'un extracteur par regles pourrait viser, ni plus
ni moins.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.paths import data_dir
from ivr_bench.generators.corpus import load_split
from ivr_bench.generators.utterances import GeneratedCase

# Arguments dont la valeur est une enumeration : ils ne sont pas presents
# litteralement dans la phrase et ne peuvent donc pas etre annotes comme
# entites. DIET les deduira de l'intention, comme le fait tout pipeline Rasa.
ENUMERATED = frozenset({"topic", "reason", "reason_category"})


def training_dir() -> Path:
    return data_dir() / "diet"


def _annotate(case: GeneratedCase) -> str:
    """Marque les valeurs d'arguments a leur position dans l'enonce."""
    text = case.utterance
    spans: list[tuple[int, int, str]] = []

    for name, value in case.expected.arguments.items():
        if name in ENUMERATED or not isinstance(value, str) or not value:
            continue
        position = text.lower().find(value.lower())
        if position < 0:
            # La valeur a ete deformee par une perturbation : on n'annote pas
            # une position approximative, ce qui apprendrait a DIET une
            # frontiere fausse.
            continue
        spans.append((position, position + len(value), name))

    # Les annotations ne doivent pas se chevaucher : la plus longue gagne.
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, str]] = []
    for start, end, name in spans:
        if all(end <= other_start or start >= other_end for other_start, other_end, _ in kept):
            kept.append((start, end, name))

    result = text
    for start, end, name in sorted(kept, key=lambda item: -item[0]):
        result = f"{result[:start]}[{result[start:end]}]({name}){result[end:]}"
    return result


def _escape(text: str) -> str:
    # Le format YAML de Rasa attend une ligne par exemple, prefixee d'un tiret.
    return re.sub(r"\s+", " ", text).strip()


def build_nlu_yaml(cases: list[GeneratedCase]) -> str:
    """Corpus d'entrainement NLU, une intention par fonction."""
    by_intent: dict[str, list[str]] = {}
    for case in cases:
        by_intent.setdefault(case.expected.tool_name, []).append(_escape(_annotate(case)))

    lines = ['version: "3.1"', "nlu:"]
    for intent in sorted(by_intent):
        lines.append(f"  - intent: {intent}")
        lines.append("    examples: |")
        for example in by_intent[intent]:
            lines.append(f"      - {example}")
    return "\n".join(lines) + "\n"


def build_config_yaml(epochs: int = 100) -> str:
    """Pipeline DIET, avec extraction d'entites (§A1, configuration 2).

    Aucun featurizer linguistique externe : la specification demande de publier
    cette variante a part, car son empreinte n'est plus comparable.
    """
    return f"""version: "3.1"
recipe: default.v1
language: fr

pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: {epochs}
    constrain_similarities: true
    entity_recognition: true
"""


def build_domain_yaml() -> str:
    catalog = default_catalog()
    entities = sorted(
        {
            parameter.name
            for definition in catalog.functions
            for parameter in definition.parameters
            if parameter.name not in ENUMERATED
        }
    )
    lines = ['version: "3.1"', "intents:"]
    lines.extend(f"  - {name}" for name in catalog.names)
    lines.append("entities:")
    lines.extend(f"  - {name}" for name in entities)
    return "\n".join(lines) + "\n"


def export(train_split: str = "train", epochs: int = 100) -> dict[str, Path]:
    """Ecrit les fichiers attendus par `rasa train nlu`."""
    directory = training_dir()
    (directory / "data").mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    nlu_path = directory / "data" / "nlu.yml"
    nlu_path.write_text(build_nlu_yaml(load_split(train_split)), encoding="utf-8")
    written["nlu"] = nlu_path

    config_path = directory / "config.yml"
    config_path.write_text(build_config_yaml(epochs=epochs), encoding="utf-8")
    written["config"] = config_path

    domain_path = directory / "domain.yml"
    domain_path.write_text(build_domain_yaml(), encoding="utf-8")
    written["domain"] = domain_path
    return written


def to_prediction(parsed: dict[str, Any]) -> tuple[str | None, dict[str, Any], float | None]:
    """Conversion deterministe intention + entites vers le schema canonique.

    C'est l'adaptateur exige par le §A1. Il ne corrige rien : si DIET n'a pas
    trouve d'entite, l'argument reste absent, comme pour toute autre
    architecture.
    """
    intent = (parsed.get("intent") or {}).get("name")
    confidence = (parsed.get("intent") or {}).get("confidence")
    if not intent:
        return None, {}, confidence

    catalog = default_catalog()
    if intent not in catalog.names:
        return None, {}, confidence

    definition = catalog.get(intent)
    if not definition.executable:
        return intent, {}, confidence

    arguments: dict[str, Any] = dict.fromkeys(definition.parameter_names)
    for entity in parsed.get("entities", []):
        name = entity.get("entity")
        if name in arguments:
            arguments[name] = entity.get("value")

    # Les arguments enumeres se deduisent de l'intention, pas du texte.
    for name in ENUMERATED:
        parameter = definition.parameter(name)
        if name in arguments and arguments[name] is None and parameter and parameter.enum:
            arguments[name] = parameter.enum[-1]
    return intent, arguments, confidence
