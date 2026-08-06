"""Export des JSON Schema derives du catalogue canonique.

Les schemas ne sont jamais ecrits a la main : ils sont regeneres depuis
`functions.yaml`, ce qui garantit qu'un argument ajoute au catalogue est
immediatement contraint dans la validation des sorties de modeles.
"""

from __future__ import annotations

import json
from pathlib import Path

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import FunctionCatalog
from ivr_bench.domain.paths import data_dir


def schemas_dir() -> Path:
    return data_dir() / "schemas"


def export_schemas(
    catalog: FunctionCatalog | None = None, destination: Path | None = None
) -> list[Path]:
    """Ecrit un schema par fonction, plus un index du catalogue."""
    catalog = catalog or default_catalog()
    target = destination or schemas_dir()
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for definition in catalog.functions:
        path = target / f"{definition.name}.schema.json"
        path.write_text(
            json.dumps(definition.json_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(path)

    index = {
        "catalog_version": catalog.version,
        "locale": catalog.locale,
        "timezone": catalog.timezone,
        "session_injected": list(catalog.session_injected),
        "functions": [
            {
                "name": definition.name,
                "executable": definition.executable,
                "schema": f"{definition.name}.schema.json",
                "required": list(definition.required_parameters),
            }
            for definition in catalog.functions
        ],
    }
    index_path = target / "catalog.index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written.append(index_path)
    return written
