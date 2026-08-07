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
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ivr_bench.domain.catalog import default_catalog
from ivr_bench.domain.models import RouterPrediction, SessionContext, ToolDefinition
from ivr_bench.domain.paths import repo_root
from ivr_bench.routers.diet.dataset import to_prediction, training_dir
from ivr_bench.routers.registry import register

SIDECAR_PYTHON = ".venv-diet/bin/python"

# Prefixe du protocole, identique cote sidecar : toute ligne qui ne le porte pas
# est une trace de Rasa, pas une reponse.
RESPONSE_PREFIX = "@@DIET@@ "


def sidecar_python() -> Path:
    return repo_root() / SIDECAR_PYTHON


def sidecar_script() -> Path:
    return Path(__file__).resolve().parent / "sidecar" / "serve.py"


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
        # Le script est lance par CHEMIN, pas comme module du paquet. L'importer
        # en tant que `ivr_bench.routers.diet.sidecar.serve` declencherait le
        # `__init__` du paquet, donc l'import de tous les routeurs et de leurs
        # dependances — pydantic, numpy, torch — qui n'existent pas dans
        # l'environnement du sidecar.
        # La sortie d'erreur part dans un FICHIER, jamais dans un tuyau. Rasa
        # journalise abondamment ; un tuyau que personne ne vide se remplit au
        # bout de quelques dizaines de kilo-octets et le sidecar se bloque en
        # ecriture, sans jamais repondre. La masquer serait pire encore : c'est
        # ainsi qu'un import manquant s'etait presente comme un « silence ».
        self._log_path = Path(tempfile.gettempdir()) / f"diet-sidecar-{os.getpid()}.log"
        self._log = self._log_path.open("w")
        self._process = subprocess.Popen(
            [str(interpreter), str(sidecar_script()), str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
        )
        self._timeout = timeout_s

        # Le chargement du modele prend plusieurs dizaines de secondes : on
        # attend le signal de disponibilite avant toute mesure de latence.
        ready = self._read_response(default="")
        if '"ready"' not in ready:
            self._process.kill()
            self._log.flush()
            details = self._log_path.read_text(encoding="utf-8", errors="replace")[-800:].strip()
            raise RuntimeError(
                "le sidecar DIET n'a pas demarre.\n"
                f"sortie : {ready.strip() or '(vide)'}\n"
                f"erreur : {details or '(vide)'}"
            )

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
        line = self._read_response()
        parsed: dict[str, Any] = json.loads(line)
        return parsed

    def _read_response(self, default: str | None = None) -> str:
        """Lit la prochaine ligne de protocole, en ignorant les traces de Rasa."""
        if self._process.stdout is None:
            raise RuntimeError("sidecar DIET indisponible")
        while True:
            line = self._process.stdout.readline()
            if not line:
                if default is not None:
                    return default
                raise RuntimeError("le sidecar DIET s'est arrete")
            if line.startswith(RESPONSE_PREFIX):
                return str(line[len(RESPONSE_PREFIX) :])

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._log.close()
        if self._process.poll() is None and self._process.stdin is not None:
            self._process.stdin.write(json.dumps({"stop": True}) + "\n")
            self._process.stdin.flush()
            self._process.wait(timeout=30)

    def __del__(self) -> None:  # pragma: no cover - filet de securite
        with contextlib.suppress(Exception):
            self.close()


register("diet")(DietRouter)
