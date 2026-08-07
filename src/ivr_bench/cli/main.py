"""Interface en ligne de commande du banc d'essai.

Les commandes suivent le paragraphe 20 de la specification. Une commande dont
l'etape n'est pas encore implementee echoue explicitement : elle ne renvoie
jamais de resultat approximatif ni de valeur de remplacement.
"""

from __future__ import annotations

from typing import NoReturn, get_args

import typer

from ivr_bench import __version__
from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.paths import repo_root, results_dir
from ivr_bench.domain.schemas import export_schemas
from ivr_bench.domain.validation import OutputValidator
from ivr_bench.generators.corpus import (
    PROFILE_VOLUMES,
    build_corpus,
    contamination_report_path,
    deduplicate,
    load_split,
    write_corpus,
)
from ivr_bench.generators.practitioners import (
    PROFILE_SIZES,
    generate_practitioners,
    write_catalog,
)
from ivr_bench.retrieval.prototypes import PrototypeStrategy

app = typer.Typer(
    name="ivr-bench",
    help="Banc d'essai de routage d'intentions pour serveur vocal francophone (CPU).",
    no_args_is_help=True,
)

data_app = typer.Typer(
    help="Generation et validation des corpus synthetiques.", no_args_is_help=True
)
doctors_app = typer.Typer(help="Catalogue synthetique de praticiens.", no_args_is_help=True)
index_app = typer.Typer(help="Construction des index semantiques.", no_args_is_help=True)
diet_app = typer.Typer(help="Pipeline Rasa DIET (sidecar Python 3.10).", no_args_is_help=True)
functiongemma_app = typer.Typer(help="FunctionGemma zero-shot et specialise.", no_args_is_help=True)
benchmark_app = typer.Typer(help="Campagnes de mesure.", no_args_is_help=True)
report_app = typer.Typer(help="Tableaux, graphiques et rapports.", no_args_is_help=True)
readme_app = typer.Typer(help="Zone generee du README.", no_args_is_help=True)
models_app = typer.Typer(help="Telechargement explicite des poids reels.", no_args_is_help=True)
domain_app = typer.Typer(help="Catalogue metier canonique.", no_args_is_help=True)

app.add_typer(domain_app, name="domain")
app.add_typer(data_app, name="data")
app.add_typer(doctors_app, name="doctors")
app.add_typer(index_app, name="index")
app.add_typer(diet_app, name="diet")
app.add_typer(functiongemma_app, name="functiongemma")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(report_app, name="report")
app.add_typer(readme_app, name="readme")
app.add_typer(models_app, name="models")


def _pending(milestone: str, what: str) -> NoReturn:
    """Signale une etape non encore implementee, sans produire de faux resultat."""
    typer.secho(
        f"{what} : etape non implementee (jalon {milestone}).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Affiche la version du banc d'essai."""
    typer.echo(__version__)


@domain_app.command("show")
def domain_show() -> None:
    """Affiche le catalogue metier canonique."""
    catalog = default_catalog()
    typer.echo(f"catalogue version {catalog.version} ({catalog.locale}, {catalog.timezone})")
    for definition in catalog.functions:
        arguments = ", ".join(definition.parameter_names) or "aucun argument"
        marker = "" if definition.executable else "  [non executable]"
        typer.echo(f"  {definition.name}{marker}")
        typer.echo(f"    arguments : {arguments}")
        if definition.required_parameters:
            typer.echo(f"    requis    : {', '.join(definition.required_parameters)}")


@domain_app.command("schemas")
def domain_schemas() -> None:
    """Regenere les JSON Schema derives du catalogue."""
    written = export_schemas()
    root = repo_root()
    for path in written:
        typer.echo(str(path.relative_to(root)))
    typer.echo(f"{len(written)} fichiers ecrits.")


@data_app.command("generate")
def data_generate(
    seed: int = typer.Option(42, help="Graine de generation."),
    profile: str = typer.Option("full", help="Profil de volumes : dev, smoke ou full."),
) -> None:
    """Genere les corpus index, train, validation et test."""
    if profile not in PROFILE_VOLUMES:
        typer.secho(
            f"profil inconnu : {profile}. Attendus : {', '.join(PROFILE_VOLUMES)}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    corpus = build_corpus(seed=seed, profile=profile)
    corpus, report = deduplicate(corpus)
    write_corpus(corpus, report, seed=seed, profile=profile)

    for split, cases in corpus.items():
        typer.echo(f"{split:12} {len(cases):6} cas")
    typer.echo(f"doublons retires : {report['total_removed']}")


@data_app.command("validate")
def data_validate(
    strict: bool = typer.Option(False, "--strict", help="Echoue si un corpus attendu est absent."),
) -> None:
    """Valide schemas, manifestes et rapport de contamination."""
    splits = ("index", "train", "validation", "contrastive", "test")
    validator = OutputValidator(default_catalog())

    missing = []
    problems = 0
    total = 0
    for split in splits:
        try:
            cases = load_split(split)
        except FileNotFoundError:
            missing.append(split)
            continue

        total += len(cases)
        for case in cases:
            # Chaque attendu doit lui-meme franchir la chaine de validation :
            # un corpus qui contiendrait une cible invalide rendrait la mesure
            # incomparable entre architectures.
            outcome = validator.validate(
                {"name": case.expected.tool_name, "arguments": case.expected.arguments}
            )
            if outcome.validity == "invalid":
                problems += 1
                if problems <= 5:
                    typer.secho(f"  {case.id} : {outcome.errors}", fg=typer.colors.RED, err=True)
        typer.echo(f"{split:12} {len(cases):6} cas valides")

    if not contamination_report_path().is_file():
        missing.append("rapport de contamination")

    if missing:
        message = f"absent : {', '.join(missing)}"
        if strict:
            typer.secho(message, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        typer.secho(message, fg=typer.colors.YELLOW, err=True)

    if problems:
        typer.secho(f"{problems} attendus invalides sur {total}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{total} cas valides au total.")


@doctors_app.command("generate")
def doctors_generate(
    seed: int = typer.Option(42, help="Graine de generation."),
    profile: str = typer.Option("full", help="Profil de volumes : dev, smoke ou full."),
) -> None:
    """Genere le catalogue synthetique de praticiens."""
    if profile not in PROFILE_SIZES:
        typer.secho(
            f"profil inconnu : {profile}. Attendus : {', '.join(PROFILE_SIZES)}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    practitioners = generate_practitioners(seed=seed, count=PROFILE_SIZES[profile])
    catalog_file, manifest_file = write_catalog(practitioners, seed=seed)
    root = repo_root()
    typer.echo(f"{len(practitioners)} praticiens ecrits dans {catalog_file.relative_to(root)}")
    typer.echo(f"manifeste : {manifest_file.relative_to(root)}")


@index_app.command("build")
def index_build(
    config: str = typer.Option("config/benchmark/cpu.yaml", help="Configuration de campagne."),
) -> None:
    """Encode les formulations et construit les prototypes."""
    import yaml

    from ivr_bench.embeddings.encoders import create_encoder
    from ivr_bench.retrieval.index import build_index

    settings = yaml.safe_load((repo_root() / config).read_text(encoding="utf-8"))
    retrieval = settings["retrieval"]

    strategy = retrieval["strategy"]
    if strategy not in get_args(PrototypeStrategy):
        typer.secho(
            f"strategie inconnue : {strategy}. "
            f"Attendues : {', '.join(get_args(PrototypeStrategy))}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    cases = load_split("index")
    encoder = create_encoder(retrieval["encoder"])
    typer.echo(f"encodage de {len(cases)} enonces avec {encoder.name} ({encoder.dimension}d)...")

    index = build_index(
        cases,
        encoder,
        strategy=strategy,
        prototypes_per_function=retrieval["prototypes_per_function"],
        seed=settings.get("seeds", [42])[0],
        batch_size=settings.get("batch_size", 32),
    )
    directory = results_dir() / "index" / f"{encoder.name}_{retrieval['strategy']}"
    index.save(directory)

    typer.echo(f"{index.size} prototypes pour {len(index.functions)} fonctions")
    typer.echo(f"index ecrit dans {directory.relative_to(repo_root())}")


@diet_app.command("train")
def diet_train(seed: int = typer.Option(42, help="Graine d'entrainement.")) -> None:
    """Entraine DIET dans le sidecar Python 3.10."""
    _pending("M7", "Entrainement DIET")


@functiongemma_app.command("train")
def functiongemma_train(seed: int = typer.Option(42, help="Graine d'entrainement.")) -> None:
    """Specialise FunctionGemma sur le corpus metier (LoRA, CPU)."""
    _pending("M7", "Specialisation de FunctionGemma")


@benchmark_app.command("text")
def benchmark_text(
    architectures: str = typer.Option("all", help="Architectures separees par des virgules."),
    config: str = typer.Option("config/benchmark/cpu.yaml", help="Configuration de campagne."),
    seed: int = typer.Option(42, help="Graine."),
    seed_count: int = typer.Option(1, help="Nombre de graines consecutives."),
    per_function: int = typer.Option(
        0,
        help="Limite de cas par fonction (0 = tout le corpus). "
        "Toute restriction est publiee dans le manifeste du run.",
    ),
) -> None:
    """Campagne texte, transcription oracle."""
    import json

    import yaml

    from ivr_bench.benchmark.runner import run_text_benchmark
    from ivr_bench.routers import available

    config_file = repo_root() / config
    settings = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    retrieval = settings.get("retrieval", {})

    names = list(available()) if architectures == "all" else architectures.split(",")
    unknown = [name for name in names if name not in available()]
    if unknown:
        typer.secho(
            f"architecture(s) inconnue(s) : {', '.join(unknown)}. "
            f"Enregistrees : {', '.join(available())}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    # Toutes les options sont proposees ; le registre ne transmet a chaque
    # architecture que celles que sa signature accepte.
    options = {
        key: retrieval[key]
        for key in ("encoder", "strategy", "top_k_prototypes", "candidates")
        if key in retrieval
    }

    for name in names:
        typer.echo(f"campagne {name}...")
        directory = run_text_benchmark(
            architecture=name,
            router_options=options,
            per_function=per_function or None,
            seed=seed,
            config_path=config_file,
            command=f"ivr-bench benchmark text --architectures {name} --seed {seed}",
        )
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        accuracy = metrics["tool_accuracy"]
        safety = metrics["emergency_handoff_recall"]
        typer.echo(
            f"  tool accuracy {accuracy:.1%} | rappel urgence "
            f"{safety:.1%} | {directory.relative_to(repo_root())}"
        )


@benchmark_app.command("audio")
def benchmark_audio(
    architectures: str = typer.Option("all", help="Architectures separees par des virgules."),
    config: str = typer.Option("config/benchmark/cpu.yaml", help="Configuration de campagne."),
    seed: int = typer.Option(42, help="Graine."),
    seed_count: int = typer.Option(1, help="Nombre de graines consecutives."),
) -> None:
    """Campagne sur transcription ASR."""
    _pending("M8", "Campagne audio")


@benchmark_app.command("e2e")
def benchmark_e2e(
    architectures: str = typer.Option("all", help="Architectures separees par des virgules."),
    config: str = typer.Option("config/benchmark/cpu.yaml", help="Configuration de campagne."),
    seed: int = typer.Option(42, help="Graine."),
    seed_count: int = typer.Option(1, help="Nombre de graines consecutives."),
) -> None:
    """Campagne bout en bout, de l'audio a l'action metier."""
    _pending("M8", "Campagne bout en bout")


@benchmark_app.command("all")
def benchmark_run_all(
    architectures: str = typer.Option("all", help="Architectures separees par des virgules."),
    config: str = typer.Option("config/benchmark/cpu.yaml", help="Configuration de campagne."),
    seed: int = typer.Option(42, help="Graine."),
    seed_count: int = typer.Option(1, help="Nombre de graines consecutives."),
) -> None:
    """Enchaine les campagnes texte, audio et bout en bout."""
    _pending("M6", "Campagne complete")


@report_app.command("build")
def report_build(
    run_id: str = typer.Option("", help="Run a traiter. Par defaut, le plus recent."),
    charts: bool = typer.Option(True, "--charts/--no-charts", help="Genere aussi les graphiques."),
) -> None:
    """Produit tableaux, graphiques et rapports."""
    _pending("M6", "Construction des rapports")


@report_app.command("verify")
def report_verify(run_id: str = typer.Option(..., help="Run a verifier.")) -> None:
    """Verifie la completude et les hashes d'un run avant publication."""
    _pending("M6", "Verification du run")


@readme_app.command("update")
def readme_update(run_id: str = typer.Option("", help="Run a publier.")) -> None:
    """Met a jour uniquement la zone generee du README."""
    from ivr_bench.reporting.readme import update

    del run_id
    if update():
        typer.echo("zone generee du README mise a jour.")
    else:
        typer.echo("zone generee deja a jour.")


@readme_app.command("check")
def readme_check() -> None:
    """Verifie la coherence de la zone generee du README."""
    from ivr_bench.reporting.readme import check

    if check():
        typer.echo("zone generee coherente.")
        return
    typer.secho(
        "zone generee du README obsolete : lancez 'ivr-bench readme update'.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)


@models_app.command("download")
def models_download(
    profile: str = typer.Option("full", help="Jeu de poids a recuperer : smoke ou full."),
) -> None:
    """Telecharge et met en cache les poids reels."""
    from ivr_bench.embeddings.download import download

    for outcome in download(profile):
        typer.echo(f"{outcome.status:4} {outcome.model_id}")


@app.command()
def reproduce(
    config: str = typer.Option("config/benchmark/cpu.yaml", help="Configuration de campagne."),
    seed: int = typer.Option(42, help="Graine."),
) -> None:
    """Rejoue la chaine complete depuis un depot propre."""
    _pending("M6", "Reproduction complete")


if __name__ == "__main__":  # pragma: no cover
    app()
