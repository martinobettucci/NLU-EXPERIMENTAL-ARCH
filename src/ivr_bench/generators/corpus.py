"""Assemblage, controle de contamination et ecriture des corpus.

La partition de test est intouchable : lorsqu'un doublon traverse la frontiere,
c'est toujours le cote entrainement qui cede. Retirer des cas du test
maquillerait la difficulte au lieu de la mesurer.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import FunctionCatalog, Practitioner
from ivr_bench.domain.paths import data_dir
from ivr_bench.generators.practitioners import load_name_banks, load_practitioners
from ivr_bench.generators.utterances import (
    GENERATOR_VERSION,
    GeneratedCase,
    Split,
    TemplateBank,
    generate_split,
    load_templates,
)

# Volumes par fonction et par partition (§10.2).
PROFILE_VOLUMES: dict[str, dict[str, int]] = {
    "full": {"index": 300, "train": 300, "validation": 100, "contrastive": 100, "test": 300},
    "smoke": {"index": 40, "train": 40, "validation": 20, "contrastive": 20, "test": 40},
    "dev": {"index": 20, "train": 20, "validation": 10, "contrastive": 10, "test": 20},
}

# Decalages de graine par partition : deux partitions du meme generateur ne
# doivent pas tirer la meme suite de valeurs.
_SEED_OFFSETS: dict[str, int] = {
    "index": 0,
    "train": 1000,
    "validation": 2000,
    "contrastive": 3000,
    "test": 7000,
}

_SPLIT_DIRECTORIES: dict[str, str] = {
    "index": "index",
    "train": "train",
    "validation": "validation",
    "contrastive": "train",
    "test": "test",
}


def generated_dir() -> Path:
    return data_dir() / "generated"


def manifest_dir() -> Path:
    return data_dir() / "manifests"


def contamination_report_path() -> Path:
    return manifest_dir() / "contamination.report.json"


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _specialties() -> list[str]:
    return list(load_name_banks()["specialties"])


def _partition_practitioners(practitioners: list[Practitioner]) -> tuple[list[Practitioner], ...]:
    """Praticiens d'entrainement d'un cote, de test de l'autre.

    Le generateur de test ne cite que des praticiens absents de l'entrainement :
    c'est ce qui distingue un routeur qui extrait un nom prononce d'un routeur
    qui a memorise une liste fermee.
    """
    train = [item for item in practitioners if item.split == "train"]
    test = [item for item in practitioners if item.split == "test"]
    if not train or not test:
        raise ValueError("le catalogue doit contenir des praticiens d'entrainement et de test")
    return train, test


def _contrastive(
    bank: TemplateBank,
    catalog: FunctionCatalog,
    practitioners: list[Practitioner],
    specialties: list[str],
    per_function: int,
    seed: int,
) -> list[GeneratedCase]:
    """Cas contrastifs : les voisins qui se confondent avec chaque fonction.

    Un cas contrastif porte l'etiquette de sa vraie fonction — celle du voisin —
    et note pour quelle fonction il constitue un piege. Il sert a durcir la
    frontiere, jamais a apprendre une mauvaise reponse.
    """
    produced: list[GeneratedCase] = []
    for definition in catalog.functions:
        neighbours = [name for name in definition.confusable_with if name != definition.name]
        if not neighbours:
            continue
        share = max(1, per_function // len(neighbours))
        for neighbour in neighbours:
            neighbour_only = FunctionCatalog(
                version=catalog.version,
                locale=catalog.locale,
                timezone=catalog.timezone,
                session_injected=catalog.session_injected,
                functions=(catalog.get(neighbour),),
            )
            batch = generate_split(
                bank,
                neighbour_only,
                practitioners,
                specialties,
                split="contrastive",
                per_function=share,
                seed=seed + hash(f"{definition.name}|{neighbour}") % 9973,
            )
            for position, case in enumerate(batch):
                produced.append(
                    case.model_copy(
                        update={
                            "id": f"contrastive_{definition.name}_{neighbour}_{position:06d}",
                            "suite": "hard_contrast",
                            "metadata": {
                                **case.metadata,
                                "contrast_for": definition.name,
                            },
                        }
                    )
                )
    return produced


def build_corpus(seed: int = 42, profile: str = "full") -> dict[str, list[GeneratedCase]]:
    """Construit toutes les partitions."""
    if profile not in PROFILE_VOLUMES:
        raise ValueError(f"profil inconnu : {profile}")

    volumes = PROFILE_VOLUMES[profile]
    catalog = default_catalog()
    specialties = _specialties()
    train_practitioners, test_practitioners = _partition_practitioners(load_practitioners())

    bank_a = load_templates("a")
    bank_b = load_templates("b")

    corpus: dict[str, list[GeneratedCase]] = {}
    training_splits: tuple[Split, ...] = ("index", "train", "validation")
    for split in training_splits:
        corpus[split] = generate_split(
            bank_a,
            catalog,
            train_practitioners,
            specialties,
            split=split,
            per_function=volumes[split],
            seed=seed + _SEED_OFFSETS[split],
        )

    corpus["contrastive"] = _contrastive(
        bank_a,
        catalog,
        train_practitioners,
        specialties,
        per_function=volumes["contrastive"],
        seed=seed + _SEED_OFFSETS["contrastive"],
    )

    corpus["test"] = generate_split(
        bank_b,
        catalog,
        test_practitioners,
        specialties,
        split="test",
        per_function=volumes["test"],
        seed=seed + _SEED_OFFSETS["test"],
    )
    return corpus


def deduplicate(
    corpus: dict[str, list[GeneratedCase]],
) -> tuple[dict[str, list[GeneratedCase]], dict[str, Any]]:
    """Retire les doublons et chiffre la contamination residuelle (§10.3)."""
    report: dict[str, Any] = {"within_split": {}, "leaked_into_training": {}, "test_size": 0}

    cleaned: dict[str, list[GeneratedCase]] = {}
    for split, cases in corpus.items():
        seen: set[str] = set()
        kept: list[GeneratedCase] = []
        for case in cases:
            key = normalized(case.utterance)
            if key in seen:
                continue
            seen.add(key)
            kept.append(case)
        cleaned[split] = kept
        report["within_split"][split] = len(cases) - len(kept)

    test_keys = {normalized(case.utterance) for case in cleaned.get("test", [])}
    report["test_size"] = len(test_keys)

    for split, cases in cleaned.items():
        if split == "test":
            continue
        # Le test ne bouge pas : c'est le cote entrainement qui cede.
        kept = [case for case in cases if normalized(case.utterance) not in test_keys]
        report["leaked_into_training"][split] = len(cases) - len(kept)
        cleaned[split] = kept

    report["total_removed"] = sum(report["within_split"].values()) + sum(
        report["leaked_into_training"].values()
    )
    return cleaned, report


def _digest(cases: list[GeneratedCase]) -> str:
    payload = "\n".join(case.model_dump_json() for case in cases)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(
    split: str, cases: list[GeneratedCase], seed: int, profile: str
) -> dict[str, Any]:
    by_function = Counter(case.expected.tool_name for case in cases)
    target = PROFILE_VOLUMES[profile].get(split, 0)
    # Une fonction qui n'atteint pas la cible est publiee comme telle : la
    # specification impose de declarer les exclusions, pas de les masquer.
    shortfall = {
        name: target - by_function.get(name, 0)
        for name in default_catalog().names
        if target and by_function.get(name, 0) < target
    }
    return {
        "split": split,
        "profile": profile,
        "seed": seed,
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "content_sha256": _digest(cases),
        "count": len(cases),
        "target_per_function": PROFILE_VOLUMES[profile].get(split, 0),
        "by_function": dict(by_function),
        "shortfall_by_function": shortfall,
        "by_suite": dict(Counter(case.suite for case in cases)),
        # Part des enonces issus directement d'un gabarit, par opposition a ceux
        # obtenus par deformation. Une part trop faible signalerait un corpus
        # dont le volume vient des bruitages plutot que des formulations.
        "by_origin": dict(Counter(str(case.metadata.get("origin", "template")) for case in cases)),
        "by_generator": dict(Counter(str(case.metadata.get("generator")) for case in cases)),
    }


def write_corpus(
    corpus: dict[str, list[GeneratedCase]],
    report: dict[str, Any],
    seed: int,
    profile: str,
) -> list[Path]:
    """Ecrit les partitions, leurs manifestes et le rapport de contamination."""
    written: list[Path] = []
    manifest_dir().mkdir(parents=True, exist_ok=True)

    for split, cases in corpus.items():
        directory = generated_dir() / _SPLIT_DIRECTORIES[split]
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(case.model_dump_json() + "\n")
        written.append(target)

        manifest = manifest_dir() / f"{split}.manifest.json"
        manifest.write_text(
            json.dumps(build_manifest(split, cases, seed, profile), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        written.append(manifest)

    report_path = contamination_report_path()
    report_path.write_text(
        json.dumps({"seed": seed, "profile": profile, **report}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    written.append(report_path)
    return written


def load_split(split: str) -> list[GeneratedCase]:
    """Relit une partition versionnee."""
    path = generated_dir() / _SPLIT_DIRECTORIES[split] / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"partition absente : {path}. Lancez 'ivr-bench data generate'.")
    with path.open(encoding="utf-8") as handle:
        return [GeneratedCase.model_validate_json(line) for line in handle if line.strip()]
