"""Generation du catalogue synthetique de praticiens (§7).

Le catalogue n'est pas une liste de noms plausibles : c'est un jeu de difficultes
choisies. Accents, apostrophes, noms composes, noms courts, noms internationaux
et surtout groupes d'homophones dont **toutes** les variantes coexistent, pour que
l'ambiguite mesuree soit reelle.

Une partition de test contient des praticiens absents de l'entrainement (§7) :
un routeur qui aurait appris une liste fermee de personnes echoue dessus, un
routeur qui extrait le nom prononce s'en sort.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ivr_bench.domain.models import Practitioner
from ivr_bench.domain.paths import data_dir
from ivr_bench.resolver.phonetics import phonetic_keys

GENERATOR_VERSION = 1

# Volumes par profil. Le profil complet respecte le minimum de 1 000 praticiens.
PROFILE_SIZES: dict[str, int] = {"dev": 120, "smoke": 200, "full": 1200}

# Part du catalogue reservee au test, donc jamais citee dans un enonce
# d'entrainement.
TEST_SPLIT_RATIO = 0.2

# Part des noms de famille volontairement portes par deux praticiens. Sans cette
# part, aucune demande par nom seul ne serait ambigue ; avec un catalogue ou tout
# est partage, aucune ne serait resolvable. Les deux cas doivent exister.
SHARED_SURNAME_RATIO = 0.08


def name_banks_path() -> Path:
    return data_dir() / "seeds" / "name_banks.yaml"


def catalog_path() -> Path:
    return data_dir() / "doctors" / "practitioners.jsonl"


def manifest_path() -> Path:
    return data_dir() / "manifests" / "practitioners.manifest.json"


def load_name_banks(path: Path | None = None) -> dict[str, Any]:
    source = path or name_banks_path()
    if not source.is_file():
        raise FileNotFoundError(f"banque de noms absente : {source}")
    with source.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"banque de noms invalide : {source}")
    return loaded


def _last_name_pool(banks: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """Noms de famille avec leur categorie et leur groupe d'homophones.

    Les groupes d'homophones sont places en tete : quel que soit le volume
    demande, toutes leurs variantes entrent dans le catalogue, sinon la mesure de
    detection d'ambiguite porterait sur des cas qui n'existent pas.
    """
    pool: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()

    for index, group in enumerate(banks.get("homophone_groups", [])):
        group_id = f"homophone_{index:02d}"
        for name in group:
            if name not in seen:
                pool.append((name, "homophone", group_id))
                seen.add(name)

    for category, names in banks.get("last_names", {}).items():
        for name in names:
            if name not in seen:
                pool.append((name, category, None))
                seen.add(name)

    return pool


def _synthetic_surnames(banks: dict[str, Any], needed: int, taken: set[str]) -> list[str]:
    """Noms de famille supplementaires, distincts et deterministes."""
    parts = banks.get("synthetic_surname_parts", {})
    prefixes: list[str] = list(parts.get("prefixes", []))
    suffixes: list[str] = list(parts.get("suffixes", []))

    produced: list[str] = []
    seen = set(taken)
    # Parcours en diagonale : les premiers noms produits ne partagent ni prefixe
    # ni suffixe, ce qui evite des blocs de noms trop ressemblants en tete.
    for offset in range(len(suffixes)):
        for index, prefix in enumerate(prefixes):
            candidate = prefix + suffixes[(index + offset) % len(suffixes)]
            if candidate in seen:
                continue
            seen.add(candidate)
            produced.append(candidate)
            if len(produced) >= needed:
                return produced
    return produced


def _test_surnames(
    selected: list[tuple[str, tuple[str, str, str | None]]], ratio: float
) -> set[str]:
    """Noms de famille reserves au test, homophones compris.

    Un groupe d'homophones part en entier du meme cote : garder `Rey` a
    l'entrainement et `Ray` au test reviendrait a avoir deja entendu le nom.
    """
    by_surname: dict[str, str | None] = {}
    for _, (last_name, _, group) in selected:
        by_surname[last_name] = group

    groups: dict[str, list[str]] = {}
    for last_name, group in by_surname.items():
        groups.setdefault(group or f"solo:{last_name}", []).append(last_name)

    # Ordre stable, independant de l'ordre d'insertion.
    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
    )

    target = int(len(by_surname) * ratio)
    chosen: set[str] = set()
    for _, surnames in ordered:
        if len(chosen) >= target:
            break
        chosen.update(surnames)
    return chosen


def generate_practitioners(seed: int = 42, count: int | None = None) -> list[Practitioner]:
    """Produit un catalogue deterministe pour une graine donnee."""
    banks = load_name_banks()
    total = count if count is not None else PROFILE_SIZES["full"]
    if total <= 0:
        raise ValueError("le catalogue doit contenir au moins un praticien")

    rng = random.Random(seed)
    first_names: list[str] = list(banks["first_names"])
    specialties: list[str] = list(banks["specialties"])
    sites: list[str] = [site["id"] for site in banks["sites"]]

    # Les noms curatifs viennent en tete, groupes d'homophones d'abord : quel que
    # soit le volume demande, les difficultes voulues entrent dans le catalogue.
    curated = _last_name_pool(banks)
    shared_count = int(total * SHARED_SURNAME_RATIO)
    distinct_needed = total - shared_count

    pool = list(curated)
    if distinct_needed > len(pool):
        pool.extend(
            (name, "synthetic", None)
            for name in _synthetic_surnames(
                banks, distinct_needed - len(pool), {item[0] for item in pool}
            )
        )
    if distinct_needed > len(pool):
        raise ValueError(
            f"{total} praticiens demandes mais seulement {len(pool)} noms de famille disponibles"
        )

    # Chaque nom retenu est porte par un seul praticien, sauf la part reservee
    # aux homonymes : l'ambiguite reste ou elle a ete choisie.
    selected: list[tuple[str, tuple[str, str, str | None]]] = []
    rotation = rng.randrange(len(first_names))
    for index, last in enumerate(pool[:distinct_needed]):
        selected.append((first_names[(rotation + index) % len(first_names)], last))
    # Les homonymes sont repartis sur tout le catalogue, pas concentres en tete :
    # sinon les noms curatifs, qui portent deja les difficultes, seraient tous
    # ambigus et plus aucun d'entre eux ne pourrait etre resolu.
    distinct = pool[:distinct_needed]
    step = max(1, len(distinct) // shared_count) if shared_count else 1
    for index, last in enumerate(distinct[::step][:shared_count]):
        # Prenom volontairement different de celui du premier porteur.
        offset = rotation + index + len(first_names) // 2 + 1
        selected.append((first_names[offset % len(first_names)], last))

    # La partition se decide par NOM DE FAMILLE, pas par position. Deux
    # praticiens homonymes doivent tomber du meme cote : sinon le nom d'un
    # praticien « de test » aurait deja ete prononce a l'entrainement par son
    # homonyme, et la partition ne prouverait plus rien.
    test_surnames = _test_surnames(selected, TEST_SPLIT_RATIO)

    practitioners: list[Practitioner] = []
    for index, (first_name, (last_name, category, group)) in enumerate(selected):
        display_name = f"Dr {first_name} {last_name}"
        aliases = (
            last_name,
            f"{first_name} {last_name}",
            f"Dr {last_name}",
            f"docteur {last_name}",
            display_name,
        )
        practitioners.append(
            Practitioner(
                practitioner_id=f"practitioner_{index + 1:05d}",
                display_name=display_name,
                first_name=first_name,
                last_name=last_name,
                specialty=specialties[index % len(specialties)],
                site_id=sites[index % len(sites)],
                aliases=aliases,
                phonetic_aliases=phonetic_keys(aliases),
                split="test" if last_name in test_surnames else "train",
                name_category=category,
                homophone_group=group,
            )
        )
    return practitioners


def _content_hash(practitioners: list[Practitioner]) -> str:
    payload = "\n".join(
        json.dumps(practitioner.model_dump(), ensure_ascii=False, sort_keys=True)
        for practitioner in practitioners
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(practitioners: list[Practitioner], seed: int) -> dict[str, Any]:
    """Manifeste du catalogue (§23)."""
    return {
        "generator": "practitioners",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        # Empreinte du contenu seul : elle ne bouge pas quand seule la date change.
        "content_sha256": _content_hash(practitioners),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "count": len(practitioners),
        "by_split": dict(Counter(item.split for item in practitioners)),
        "by_name_category": dict(Counter(item.name_category for item in practitioners)),
        "by_specialty": dict(Counter(item.specialty for item in practitioners)),
        "by_site": dict(Counter(item.site_id for item in practitioners)),
        "homophone_groups": dict(
            Counter(item.homophone_group for item in practitioners if item.homophone_group)
        ),
    }


def write_catalog(
    practitioners: list[Practitioner],
    seed: int,
    destination: Path | None = None,
    manifest_destination: Path | None = None,
) -> tuple[Path, Path]:
    """Ecrit le catalogue en JSONL et son manifeste."""
    target = destination or catalog_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for practitioner in practitioners:
            handle.write(
                json.dumps(practitioner.model_dump(), ensure_ascii=False, sort_keys=True) + "\n"
            )

    manifest_target = manifest_destination or manifest_path()
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_text(
        json.dumps(build_manifest(practitioners, seed), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target, manifest_target


def load_practitioners(path: Path | None = None) -> list[Practitioner]:
    """Relit le catalogue versionne."""
    source = path or catalog_path()
    if not source.is_file():
        raise FileNotFoundError(
            f"catalogue de praticiens absent : {source}. Lancez 'ivr-bench doctors generate'."
        )
    with source.open(encoding="utf-8") as handle:
        return [Practitioner.model_validate_json(line) for line in handle if line.strip()]
