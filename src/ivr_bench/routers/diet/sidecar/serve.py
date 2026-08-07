"""Service DIET, execute par l'interpreteur Python 3.10 du sidecar.

Rasa 3.6 declare `requires_python <3.11` : il ne peut pas cohabiter avec
l'environnement principal. Plutot que de reimplementer DIET — ce qui ne
prouverait rien sur Rasa — on l'isole ici et on lui parle par un protocole
ligne a ligne : une requete JSON par ligne sur l'entree, une reponse JSON par
ligne sur la sortie.

Le modele est charge une seule fois : le recharger a chaque phrase melangerait
le cout de demarrage a la latence d'inference.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


def main() -> int:
    model_path = Path(sys.argv[1])
    from rasa.core.agent import Agent

    agent = Agent.load(str(model_path))
    loop = asyncio.new_event_loop()

    # Signale au parent que le modele est pret avant d'attendre des requetes.
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "requete illisible"}), flush=True)
            continue
        if request.get("stop"):
            break
        try:
            parsed = loop.run_until_complete(agent.parse_message(request.get("text", "")))
            print(json.dumps(parsed, default=str), flush=True)
        except Exception as error:
            print(json.dumps({"error": str(error)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
