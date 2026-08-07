"""A1 — Rasa DIET, pilote depuis un sidecar Python 3.10.

Rasa 3.6 exige Python < 3.11. Plutot que de reimplementer DIET — ce qui ne
dirait rien de Rasa — le vrai pipeline tourne dans son propre environnement et
recoit les phrases par un protocole ligne a ligne. L'adaptateur convertit
ensuite intention et entites vers le schema de fonction canonique, de facon
deterministe (§A1).
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import RouterPrediction, SessionContext, ToolDefinition
from ivr_bench.domain.paths import repo_root
from ivr_bench.routers.diet.dataset import to_prediction, training_dir
from ivr_bench.routers.registry import register

SIDECAR_PYTHON = ".venv-diet/bin/python"


def sidecar_python() -> Path:
    return repo_root() / SIDECAR_PYTHON


def latest_model() -> Path:
    """Dernier modele entraine par le sidecar."""
    models = sorted((training_dir() / "models").glob("*.tar.gz"))
    if not models:
        raise FileNotFoundError("aucun modele DIET entraine. Lancez 'ivr-bench diet train'.")
    return models[-1]


class DietRouter:
    """Adaptateur vers le pipeline NLU de Rasa."""

    name = "diet"

    def __init__(self, model_path: str | None = None, timeout_s: float = 120.0) -> None:
        self._catalog = default_catalog()
        interpreter = sidecar_python()
        if not interpreter.is_file():
            raise RuntimeError(
                "environnement DIET absent. Creez-le avec "
                "'python3.10 -m venv .venv-diet && .venv-diet/bin/pip install rasa==3.6.21'."
            )

        model = Path(model_path) if model_path else latest_model()
        self._process = subprocess.Popen(
            [
                str(interpreter),
                "-m",
                "ivr_bench.routers.diet.sidecar.serve",
                str(model),
            ],
            cwd=repo_root() / "src",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._timeout = timeout_s

        # Le chargement du modele prend plusieurs dizaines de secondes : on
        # attend le signal de disponibilite avant toute mesure de latence.
        ready = self._process.stdout.readline() if self._process.stdout else ""
        if '"ready"' not in ready:
            raise RuntimeError(f"le sidecar DIET n'a pas demarre : {ready.strip() or 'silence'}")

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        started = time.perf_counter()
        allowed = {tool.name for tool in tools} or set(self._catalog.names)

        parsed = self._ask(utterance)
        name, arguments, confidence = to_prediction(parsed)
        if name is not None and name not in allowed:
            name, arguments = (None, {})

        return RouterPrediction(
            tool_name=name,
            arguments=arguments,
            confidence=round(confidence, 4) if confidence is not None else None,
            raw_output=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "router": self.name,
                "intent": (parsed.get("intent") or {}).get("name"),
                "entities": len(parsed.get("entities", [])),
            },
        )

    def _ask(self, utterance: str) -> dict[str, Any]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("sidecar DIET indisponible")
        self._process.stdin.write(json.dumps({"text": utterance}) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("le sidecar DIET s'est arrete")
        parsed: dict[str, Any] = json.loads(line)
        return parsed

    def close(self) -> None:
        if self._process.poll() is None and self._process.stdin is not None:
            self._process.stdin.write(json.dumps({"stop": True}) + "\n")
            self._process.stdin.flush()
            self._process.wait(timeout=30)

    def __del__(self) -> None:  # pragma: no cover - filet de securite
        with contextlib.suppress(Exception):
            self.close()


register("diet")(DietRouter)
