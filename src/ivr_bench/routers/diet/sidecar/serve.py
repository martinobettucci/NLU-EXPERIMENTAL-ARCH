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
import logging
import sys
from pathlib import Path

# Marqueur de protocole. Rasa journalise abondamment, et rien ne garantit que
# ses lignes n'atterrissent pas sur la sortie standard : sans marqueur, une
# ligne de debogage serait lue comme une reponse.
RESPONSE_PREFIX = "@@DIET@@ "


def _emit(payload: dict[str, object]) -> None:
    print(RESPONSE_PREFIX + json.dumps(payload, default=str), flush=True)


def main() -> int:
    model_path = Path(sys.argv[1])

    # Toute la journalisation part vers l'erreur standard : la sortie standard
    # est reservee au protocole.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)
    for name in ("rasa", "tensorflow", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)

    from rasa.core.agent import Agent

    agent = Agent.load(str(model_path))
    loop = asyncio.new_event_loop()
    # Rasa interroge la boucle courante : sans cet enregistrement, certaines de
    # ses coroutines s'attachent a une autre boucle et n'aboutissent jamais.
    asyncio.set_event_loop(loop)

    # Signale au parent que le modele est pret avant d'attendre des requetes.
    _emit({"ready": True})

    # `for line in sys.stdin` lit par blocs : tant que le tampon interne n'est
    # pas plein, aucune ligne n'est rendue et le sidecar reste muet alors que le
    # parent attend sa reponse. Un test ou l'entree se referme aussitot ne
    # revele pas le probleme — l'EOF vide le tampon. En service continu, il
    # bloque indefiniment. On lit donc ligne a ligne, explicitement.
    while True:
        raw = sys.stdin.readline()
        if not raw:
            break
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _emit({"error": "requete illisible"})
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
