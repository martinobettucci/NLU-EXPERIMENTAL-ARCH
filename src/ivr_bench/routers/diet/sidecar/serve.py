"""Service DIET, execute par l'interpreteur Python 3.10 du sidecar.

Rasa 3.6 exige Python < 3.11 et ne peut pas cohabiter avec l'environnement
principal. Le dialogue se fait donc **par fichiers**, pas par tuyaux.

Ce choix vient de l'experience : quatre tentatives de protocole interactif ont
echoue de quatre facons differentes — sortie d'erreur non vidangee, lecture
anticipee de stdin, marqueur colle a une trace sans retour a la ligne, sortie
reconfiguree par TensorFlow. Chacune produisait le meme symptome : deux
processus qui s'attendent. Un fichier d'entree et un fichier de sortie
suppriment la classe entiere de ces pannes, puisque rien ne peut se bloquer sur
un tampon.

La latence de chaque analyse est mesuree ici, au plus pres du modele, et
transmise avec la reponse. Elle exclut donc le cout de transport, ce qui la
rend plus juste que ce qu'un protocole interactif aurait mesure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: serve.py <modele> <requetes.jsonl> <reponses.jsonl>", file=sys.stderr)
        return 2

    model_path, requests_path, responses_path = (Path(argument) for argument in sys.argv[1:4])

    # Toute la journalisation part vers l'erreur standard, redirigee vers un
    # fichier par l'appelant.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)

    from rasa.core.agent import Agent

    agent = Agent.load(str(model_path))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    with (
        requests_path.open(encoding="utf-8") as source,
        responses_path.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            request = json.loads(stripped)
            text = request.get("text", "")
            started = time.perf_counter()
            try:
                parsed = loop.run_until_complete(agent.parse_message(text))
            except Exception as error:
                parsed = {"error": str(error)}
            parsed["latency_ms"] = (time.perf_counter() - started) * 1000.0
            target.write(json.dumps(parsed, default=str) + "\n")
            target.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
