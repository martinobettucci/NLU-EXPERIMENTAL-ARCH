"""A1 — Rasa DIET, pilote depuis un sidecar Python 3.10.

Rasa 3.6 exige Python < 3.11. Plutot que de reimplementer DIET — ce qui ne
dirait rien de Rasa — le vrai pipeline tourne dans son propre environnement.

Le dialogue se fait par fichiers et en lot, pas par tuyaux interactifs. Voir
`sidecar/serve.py` pour la raison : quatre protocoles interactifs successifs se
sont bloques, chacun pour un motif different. Le mode fichier supprime la cause
commune plutot que de la traiter symptome par symptome.

Consequence sur le contrat : ce routeur expose `prepare()`, appele par le
harnais avant la boucle de mesure. Un routeur qui ne l'expose pas n'est pas
concerne.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
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

    def __init__(self, model_path: str | None = None) -> None:
        self._catalog = default_catalog()
        self._model = Path(model_path) if model_path else latest_model()
        self._parsed: dict[str, dict[str, Any]] = {}

        interpreter = sidecar_python()
        if not interpreter.is_file():
            raise RuntimeError(
                "environnement DIET absent. Creez-le avec "
                "'python3.10 -m venv .venv-diet && .venv-diet/bin/pip install rasa==3.6.21'."
            )

    def prepare(self, utterances: list[str]) -> None:
        """Analyse tout le lot en une passe, modele charge une seule fois.

        Le chargement du modele TensorFlow coute plusieurs minutes ; le payer
        une fois par campagne plutot qu'une fois par phrase est la seule facon
        d'obtenir une latence d'inference interpretable.
        """
        unique = list(dict.fromkeys(utterances))
        if not unique:
            return

        with tempfile.TemporaryDirectory() as workspace:
            directory = Path(workspace)
            requests = directory / "requests.jsonl"
            responses = directory / "responses.jsonl"
            log = directory / "sidecar.log"

            requests.write_text(
                "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in unique),
                encoding="utf-8",
            )

            with log.open("w") as errors:
                completed = subprocess.run(
                    [
                        str(sidecar_python()),
                        str(sidecar_script()),
                        str(self._model),
                        str(requests),
                        str(responses),
                    ],
                    stdout=errors,
                    stderr=errors,
                    check=False,
                )

            if completed.returncode != 0 or not responses.is_file():
                details = log.read_text(encoding="utf-8", errors="replace")[-800:].strip()
                raise RuntimeError(f"le sidecar DIET a echoue :\n{details or '(aucune trace)'}")

            lines = [line for line in responses.read_text(encoding="utf-8").splitlines() if line]
            if len(lines) != len(unique):
                raise RuntimeError(
                    f"{len(lines)} reponses pour {len(unique)} requetes : lot incomplet"
                )
            self._parsed = {
                text: json.loads(line) for text, line in zip(unique, lines, strict=True)
            }

    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        parsed = self._parsed.get(utterance)
        if parsed is None:
            # Le lot n'a pas ete prepare : on echoue plutot que de charger le
            # modele phrase par phrase, ce qui produirait une latence n'ayant
            # aucun rapport avec l'inference.
            raise RuntimeError("enonce absent du lot analyse : appelez prepare() avant predict().")

        allowed = {tool.name for tool in tools} or set(self._catalog.names)
        name, arguments, confidence = to_prediction(parsed)
        if name is not None and name not in allowed:
            name, arguments = (None, {})

        return RouterPrediction(
            tool_name=name,
            arguments=arguments,
            confidence=round(confidence, 4) if confidence is not None else None,
            raw_output=None,
            # Latence mesuree dans le sidecar, au plus pres du modele.
            latency_ms=float(parsed.get("latency_ms", 0.0)),
            metadata={
                "router": self.name,
                "intent": (parsed.get("intent") or {}).get("name"),
                "entities": len(parsed.get("entities", [])),
                "batched": True,
            },
        )


register("diet")(DietRouter)
