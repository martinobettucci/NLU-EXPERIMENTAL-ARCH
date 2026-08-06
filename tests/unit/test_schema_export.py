"""Les JSON Schema versionnes doivent rester derives du catalogue canonique."""

from __future__ import annotations

import json
from pathlib import Path

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.schemas import export_schemas, schemas_dir


def test_versioned_schemas_match_the_catalog(tmp_path: Path) -> None:
    """Un argument ajoute au catalogue sans regeneration doit faire echouer la CI."""
    export_schemas(destination=tmp_path)

    for regenerated in sorted(tmp_path.glob("*.json")):
        versioned = schemas_dir() / regenerated.name
        assert versioned.is_file(), (
            f"{regenerated.name} manque dans data/schemas : lancez 'ivr-bench domain schemas'."
        )
        assert json.loads(versioned.read_text(encoding="utf-8")) == json.loads(
            regenerated.read_text(encoding="utf-8")
        ), f"{regenerated.name} n'est plus synchronise avec functions.yaml."


def test_schema_rejects_unknown_arguments() -> None:
    schema = default_catalog().get("list_appointments").json_schema()
    assert schema["additionalProperties"] is False


def test_nullable_argument_accepts_null_in_schema() -> None:
    schema = default_catalog().get("list_appointments").json_schema()
    assert schema["properties"]["date_from"]["type"] == ["string", "null"]
    assert schema["required"] == []
