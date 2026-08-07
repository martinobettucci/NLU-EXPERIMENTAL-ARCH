"""Tests des tableaux, des graphiques et de la zone generee du README."""

from __future__ import annotations

from pathlib import Path

import pytest

from ivr_bench.reporting.readme import END, NOT_RUN, START, RunSummary, render


def _summary(architecture: str, dirty: bool = False, **metrics: object) -> RunSummary:
    base = {
        "architecture": architecture,
        "tool_accuracy": 0.5,
        "macro_f1": 0.4,
        "emergency_handoff_recall": 0.9,
        "no_tool_recall": 0.3,
        "argument_exact_match": 0.7,
        "hallucinated_argument_rate": 0.1,
        "latency_ms": {"p50": 10.0, "p95": 20.0},
        "count": 84,
        "coverage": {"evaluated": 84, "available": 2088, "restricted": True},
    }
    base.update(metrics)
    environment = {
        "git_dirty": dirty,
        "timestamp": "2026-03-02T09:00:00+00:00",
        "git_commit": "abcdef123456789",
        "cpu": "x86_64",
        "cpu_threads": 4,
        "ram_gb": 15.7,
        "gpu": "aucun",
    }
    return RunSummary(architecture, base, environment)


def test_missing_architecture_reads_not_run() -> None:
    rendered = render({})
    assert rendered.count(NOT_RUN) >= 9 * 7
    assert "0.0%" not in rendered


def test_a_dirty_run_is_never_published() -> None:
    """Un depot modifie rend le run irreproductible (§28)."""
    assert not _summary("rules", dirty=True).is_publishable
    assert _summary("rules", dirty=False).is_publishable


def test_partial_coverage_is_declared_in_the_table() -> None:
    rendered = render({"needle_full": _summary("needle_full")})
    assert "84 / 2088" in rendered
    assert "Couverture partielle" in rendered


def test_full_coverage_shows_a_plain_count() -> None:
    summary = _summary(
        "rules", coverage={"evaluated": 2088, "available": 2088, "restricted": False}
    )
    rendered = render({"rules": summary})
    assert "| 2088 |" in rendered
    assert "Couverture partielle" not in rendered


def test_a_zero_measurement_is_shown_as_zero_not_hidden() -> None:
    """Un vrai zero mesure doit apparaitre : c'est un resultat, pas une absence."""
    rendered = render({"needle_full": _summary("needle_full", emergency_handoff_recall=0.0)})
    assert "0.0%" in rendered


def test_markers_are_preserved_on_update(tmp_path: Path) -> None:
    from ivr_bench.reporting.readme import update

    target = tmp_path / "README.md"
    target.write_text(f"# Titre\n\navant\n\n{START}\nvieux\n{END}\n\napres\n", encoding="utf-8")
    update(target)
    written = target.read_text(encoding="utf-8")
    assert written.startswith("# Titre\n\navant\n")
    assert written.rstrip().endswith("apres")
    assert "vieux" not in written


def test_update_without_markers_is_rejected(tmp_path: Path) -> None:
    from ivr_bench.reporting.readme import update

    target = tmp_path / "README.md"
    target.write_text("# Sans marqueurs\n", encoding="utf-8")
    with pytest.raises(ValueError, match="marqueurs"):
        update(target)


def test_summary_table_is_written_with_its_sources() -> None:
    """Une figure sans ses chiffres n'est pas verifiable (§31)."""
    from ivr_bench.reporting.tables import tables_dir

    directory = tables_dir()
    if not (directory / "summary.csv").is_file():
        pytest.skip("aucun tableau genere : lancez 'ivr-bench report build'")
    for name in ("summary", "by_suite", "by_function"):
        assert (directory / f"{name}.csv").is_file()
        assert (directory / f"{name}.parquet").is_file()


def test_paired_comparisons_only_use_shared_cases() -> None:
    from ivr_bench.reporting.tables import tables_dir

    path = tables_dir() / "paired_comparisons.csv"
    if not path.is_file():
        pytest.skip("aucune comparaison generee")
    import csv

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        # Comparer 84 cas a 2088 produirait un classement sans signification.
        assert int(row["shared_cases"]) >= 20
