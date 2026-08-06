"""Interface en ligne de commande du banc d'essai.

Les commandes suivent le paragraphe 20 de la specification. Une commande dont
l'etape n'est pas encore implementee echoue explicitement : elle ne renvoie
jamais de resultat approximatif ni de valeur de remplacement.
"""

from __future__ import annotations

from typing import NoReturn

import typer

from ivr_bench import __version__
from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.paths import repo_root
from ivr_bench.domain.schemas import export_schemas
from ivr_bench.generators.practitioners import (
    PROFILE_SIZES,
    generate_practitioners,
    write_catalog,
)

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
    _pending("M3", "Generation des corpus")


@data_app.command("validate")
def data_validate(
    strict: bool = typer.Option(False, "--strict", help="Echoue si un corpus attendu est absent."),
) -> None:
    """Valide schemas, manifestes et rapport de contamination."""
    _pending("M3", "Validation des corpus")


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
    _pending("M4", "Construction de l'index")


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
) -> None:
    """Campagne texte, transcription oracle."""
    _pending("M6", "Campagne texte")


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
    _pending("M6", "Mise a jour du README")


@readme_app.command("check")
def readme_check() -> None:
    """Verifie la coherence de la zone generee du README."""
    _pending("M6", "Verification du README")


@models_app.command("download")
def models_download(
    profile: str = typer.Option("full", help="Jeu de poids a recuperer : smoke ou full."),
) -> None:
    """Telecharge et met en cache les poids reels."""
    _pending("M4", "Telechargement des poids")


@app.command()
def reproduce(
    config: str = typer.Option("config/benchmark/cpu.yaml", help="Configuration de campagne."),
    seed: int = typer.Option(42, help="Graine."),
) -> None:
    """Rejoue la chaine complete depuis un depot propre."""
    _pending("M6", "Reproduction complete")


if __name__ == "__main__":  # pragma: no cover
    app()
