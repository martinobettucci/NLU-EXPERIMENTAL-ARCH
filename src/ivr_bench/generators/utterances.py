"""Generation des corpus synthetiques (§10).

Deux generateurs independants produisent des partitions disjointes par
construction : A alimente l'index, l'entrainement et la validation ; B produit le
test, avec d'autres familles de gabarits, un autre lexique et des praticiens
issus de la partition de test. La deduplication n'est qu'un second rideau : elle
mesure la contamination residuelle, elle ne la cree pas.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ivr_bench.domain.models import FunctionCatalog, Practitioner
from ivr_bench.domain.paths import data_dir
from ivr_bench.generators.perturbations import PERTURBATIONS, apply
from ivr_bench.resolver.phonetics import strip_accents

GENERATOR_VERSION = 1

Split = Literal["index", "train", "validation", "test", "contrastive"]

# Correspondance entre un emplacement de gabarit et l'argument attendu, par
# fonction. Le meme « mardi » devient `preferred_date` pour une prise de
# rendez-vous et `preferred_new_date` pour une modification.
SLOT_ARGUMENTS: dict[str, dict[str, str]] = {
    "request_new_appointment": {
        "practitioner": "practitioner_name",
        "specialty": "specialty",
        "date": "preferred_date",
        "time": "preferred_time",
    },
    "request_appointment_reschedule": {
        "practitioner": "practitioner_name",
        "date": "preferred_new_date",
        "time": "preferred_new_time",
    },
    "list_appointments": {
        "practitioner": "practitioner_name",
        "date": "date_from",
    },
}

# Nombre maximal d'enonces partageant le meme noyau semantique. Sans ce plafond,
# une meme question decline uniquement par sa formule de politesse remplirait
# l'index de quasi-doublons et gonflerait artificiellement les scores du
# retriever : les §10.3 exige une deduplication semantique, pas seulement
# lexicale.
MAX_PER_SEMANTIC_CORE = 2

# Nombre d'echecs consecutifs apres lequel une deformation est consideree comme
# epuisee pour cette fonction.
_STERILE_STREAK = 40

# Mots vides retires pour calculer ce noyau : politesses, hesitations et
# connecteurs qui ne changent pas l'intention.
_CORE_STOPWORDS = frozenset(
    {
        "s",
        "il",
        "vous",
        "plait",
        "merci",
        "si",
        "ca",
        "va",
        "voila",
        "euh",
        "hum",
        "alors",
        "en",
        "fait",
        "ben",
        "le",
        "la",
        "les",
        "un",
        "une",
        "de",
        "du",
        "des",
        "a",
        "au",
        "aux",
        "que",
        "qui",
        "est",
        "ce",
        "je",
        "j",
        "me",
        "mon",
        "ma",
        "mes",
        "et",
        "d",
        "l",
        "pour",
    }
)

# Formes sous lesquelles un praticien peut etre nomme a l'oral.
_PRACTITIONER_FORMS = (
    "docteur {last}",
    "le docteur {last}",
    "Dr {last}",
    "docteur {first} {last}",
    "madame le docteur {last}",
    "professeur {last}",
)


class ExpectedCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class GeneratedCase(BaseModel):
    """Un cas du corpus, au format JSONL versionne (§23)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    split: Split
    suite: str
    language: str = "fr"
    utterance: str
    expected: ExpectedCall
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemplateFamily(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    function: str
    suite: str
    patterns: tuple[str, ...]
    slots: tuple[str, ...] = ()
    topic: str | None = None
    reason: str | None = None
    reason_category: str | None = None


class TemplateBank(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    generator: str
    lexicon: dict[str, tuple[str, ...]]
    families: tuple[TemplateFamily, ...]


def templates_path(generator: str) -> Path:
    return data_dir() / "seeds" / f"templates_{generator}.yaml"


def load_templates(generator: str, path: Path | None = None) -> TemplateBank:
    source = path or templates_path(generator)
    if not source.is_file():
        raise FileNotFoundError(f"gabarits absents : {source}")
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TemplateBank.model_validate(raw)


def semantic_core(text: str) -> str:
    """Noyau d'un enonce : ce qu'il reste une fois les formules retirees."""
    words = strip_accents(text.lower()).replace("'", " ").split()
    kept = [word.strip(".,;:!?") for word in words]
    return " ".join(sorted(word for word in kept if word and word not in _CORE_STOPWORDS))


def _practitioner_form(practitioner: Practitioner, rng: random.Random) -> str:
    template = rng.choice(_PRACTITIONER_FORMS)
    return template.format(first=practitioner.first_name, last=practitioner.last_name)


def _fill(
    family: TemplateFamily,
    pattern: str,
    catalog: FunctionCatalog,
    bank: TemplateBank,
    practitioners: list[Practitioner],
    specialties: list[str],
    rng: random.Random,
) -> tuple[str, dict[str, Any]]:
    """Instancie un gabarit et deduit les arguments attendus."""
    mapping = SLOT_ARGUMENTS.get(family.function, {})
    definition = catalog.get(family.function)

    # Tout argument de la fonction est present dans l'attendu ; ceux que l'enonce
    # n'exprime pas valent explicitement `null`. Le routeur ne doit pas les
    # inventer, et la metrique doit pouvoir distinguer absent de hallucine (§12).
    arguments: dict[str, Any] = dict.fromkeys(definition.parameter_names)

    values: dict[str, str] = {
        "opener": rng.choice(bank.lexicon.get("openers", ("",))),
        "politeness": rng.choice(bank.lexicon.get("politeness", ("",))),
    }

    for slot in family.slots:
        if slot == "practitioner":
            spoken = _practitioner_form(rng.choice(practitioners), rng)
        elif slot == "specialty":
            spoken = rng.choice(specialties)
        elif slot == "date":
            spoken = rng.choice(bank.lexicon.get("dates", ("demain",)))
        elif slot == "time":
            spoken = rng.choice(bank.lexicon.get("times", ("le matin",)))
        else:
            raise ValueError(f"emplacement inconnu : {slot}")
        values[slot] = spoken
        argument = mapping.get(slot)
        if argument and argument in arguments:
            arguments[argument] = spoken

    if family.topic is not None:
        arguments["topic"] = family.topic
    if family.reason is not None:
        arguments["reason"] = family.reason
    if family.reason_category is not None:
        arguments["reason_category"] = family.reason_category

    text = pattern
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    text = " ".join(text.split()).replace(" ?", " ?").strip()

    if not definition.executable:
        arguments = {}
    return text, arguments


def _base_cases(
    bank: TemplateBank,
    catalog: FunctionCatalog,
    practitioners: list[Practitioner],
    specialties: list[str],
    rng: random.Random,
    attempts_per_pattern: int,
) -> dict[str, list[tuple[str, str, dict[str, Any], str]]]:
    """Cas de base par fonction : (enonce, suite, arguments, famille)."""
    produced: dict[str, list[tuple[str, str, dict[str, Any], str]]] = {}
    known = set(catalog.names)
    for family in bank.families:
        # La generation contrastive ne demande qu'une fonction a la fois : les
        # familles des autres fonctions ne la concernent pas.
        if family.function not in known:
            continue
        for pattern in family.patterns:
            for _ in range(attempts_per_pattern):
                text, arguments = _fill(
                    family, pattern, catalog, bank, practitioners, specialties, rng
                )
                produced.setdefault(family.function, []).append(
                    (text, family.suite, arguments, family.id)
                )
    return produced


def generate_split(
    bank: TemplateBank,
    catalog: FunctionCatalog,
    practitioners: list[Practitioner],
    specialties: list[str],
    split: Split,
    per_function: int,
    seed: int,
) -> list[GeneratedCase]:
    """Produit `per_function` cas distincts pour chaque fonction du catalogue."""
    rng = random.Random(seed)
    # Assez de tirages pour que les emplacements varient avant de recourir aux
    # deformations, qui ne doivent pas devenir la source principale de volume.
    pool = _base_cases(
        bank,
        catalog,
        practitioners,
        specialties,
        rng,
        attempts_per_pattern=max(8, per_function // 4),
    )

    cases: list[GeneratedCase] = []
    for function in catalog.names:
        entries = pool.get(function, [])
        if not entries:
            raise ValueError(f"aucun gabarit pour la fonction {function} dans {bank.generator}")

        # Le vivier est parcouru melange : sans cela, les premiers gabarits
        # fourniraient a eux seuls tout le volume et les dernieres familles
        # n'apparaitraient jamais.
        shuffled = list(entries)
        rng.shuffle(shuffled)

        seen: set[str] = set()
        cores: Counter[str] = Counter()
        selected: list[tuple[str, str, dict[str, Any], str]] = []
        for text, suite, arguments, family_id in shuffled:
            key = " ".join(text.lower().split())
            core = semantic_core(text)
            if key in seen or cores[core] >= MAX_PER_SEMANTIC_CORE:
                continue
            seen.add(key)
            cores[core] += 1
            selected.append((text, suite, arguments, family_id))
            if len(selected) >= per_function:
                break

        # Complement par deformations : hesitations, fautes, erreurs de
        # transcription. Les trois sont tirees a tour de role avec le meme quota :
        # sans cela, la deformation qui produit le plus facilement une chaine
        # inedite - la faute de frappe - representerait a elle seule la moitie du
        # corpus, et la degradation mesuree ne dirait plus rien de la parole.
        kinds = [kind for kinds in PERTURBATIONS.values() for kind in kinds]
        suite_of = {kind: name for name, values in PERTURBATIONS.items() for kind in values}
        produced_by_kind: Counter[str] = Counter()

        # Le plafond par noyau semantique ne s'applique PAS ici. Il vise les
        # quasi-doublons de remplissage - la meme question redite avec une autre
        # formule de politesse - alors qu'une deformation est censee conserver le
        # sens : c'est tout son interet. L'appliquer aux deformations
        # etoufferait hesitations et erreurs de transcription, dont le noyau est
        # justement inchange, et laisserait les fautes de frappe occuper seules
        # le corpus.
        used: set[tuple[int, str]] = set()
        candidates = [(position, kind) for kind in kinds for position in range(len(entries))]
        rng.shuffle(candidates)
        # Tri stable par quota : les trois deformations progressent de front.
        for position, kind in sorted(candidates, key=lambda item: kinds.index(item[1])):
            if len(selected) >= per_function:
                break
            if (position, kind) in used:
                continue
            used.add((position, kind))
            source = entries[position]
            text = apply(kind, source[0], rng)
            key = " ".join(text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            selected.append((text, suite_of[kind], source[2], source[3]))
            produced_by_kind[kind] += 1

        base_count = len(selected) - sum(produced_by_kind.values())
        for position, (text, suite, arguments, family_id) in enumerate(selected):
            cases.append(
                GeneratedCase(
                    id=f"{split}_{function}_{position:06d}",
                    split=split,
                    suite=suite,
                    utterance=text,
                    expected=ExpectedCall(tool_name=function, arguments=arguments),
                    metadata={
                        "generator": bank.generator,
                        "template_family": family_id,
                        # Tout est synthetique : aucune donnee reelle n'entre ici.
                        "origin": "template" if position < base_count else "perturbation",
                        "contains_sensitive_synthetic_data": False,
                    },
                )
            )

    return cases
