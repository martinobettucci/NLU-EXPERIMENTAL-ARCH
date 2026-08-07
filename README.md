# NLU-EXPERIMENTAL-ARCH

Banc d'essai reproductible pour le routage d'intentions et l'appel de fonctions dans un
serveur vocal interactif francophone.

Le dépôt compare des architectures capables de transformer une demande vocale en une fonction
métier structurée, dans le cadre d'une plateforme de prise de rendez-vous médicaux. Ce n'est
**pas** un assistant médical : aucune question clinique n'est traitée, elle est refusée ou
transférée selon une politique déterministe implémentée hors modèle.

**Tout s'exécute sur CPU.** C'est la prémisse de l'expérience : on compare des architectures
déployables sur du matériel ordinaire ou embarqué, pas des modèles adossés à un accélérateur.

## Hypothèse testée

> Une architecture hybride qui précompile des formulations hypothétiques par fonction, les
> encode dans un espace vectoriel compact, récupère deux fonctions candidates au runtime, puis
> confie uniquement ces fonctions à un micro-modèle d'appel d'outils peut atteindre une
> précision comparable ou supérieure à un appel direct sur le catalogue complet, avec une
> latence, une mémoire et un coût d'inférence inférieurs.

Le protocole complet est décrit dans
[`docs/SPECIFICATION_IVR_ROUTING_BENCHMARK.md`](docs/SPECIFICATION_IVR_ROUTING_BENCHMARK.md).

## Architectures comparées

| Id | Architecture | Modèle |
|----|--------------|--------|
| A0 | Baseline déterministe | règles et expressions régulières |
| A1 | DIET | Rasa 3.6 (sidecar Python 3.10) |
| A2 | Needle, catalogue complet | `Cactus-Compute/needle`, 26M |
| A3 | FunctionGemma zéro-shot | `google/functiongemma-270m-it` |
| A4 | FunctionGemma spécialisé | idem, LoRA sur le corpus métier |
| A5 | Retriever sémantique seul | EmbeddingGemma 300M, MRL |
| A6 | Retriever top 2 puis Needle | architecture proposée |
| A7 | Retriever top 2 puis FunctionGemma | variante |
| A8 | Sélection adaptative | 1 à 4 fonctions selon l'incertitude |

## Démarrage

```bash
make setup                # environnement complet, CPU uniquement
make test                 # tests ne nécessitant aucun poids
export HF_TOKEN=...       # EmbeddingGemma et FunctionGemma sont sous licence Gemma
make models-download      # téléchargement explicite des poids
make generate-data        # corpus synthétiques et catalogue de praticiens
make build-index          # embeddings et prototypes
make benchmark-smoke      # campagne courte sur modèles réels
```

`./runDev`, `./runBenchmark` et `./runProd` correspondent aux trois profils décrits au §34 de
la spécification. `runProd` ne lance que la démonstration du serveur vocal simulé : le dépôt
reste avant tout un projet expérimental.

## Principes de rigueur

- **Aucun substitut.** Pas d'encodeur factice ni de routeur simulé, y compris en intégration
  continue. Un poids manquant fait échouer la suite au lieu d'être remplacé.
- **Aucun résultat saisi à la main.** Toute métrique du tableau ci-dessous provient d'un run
  tracé dans `results/runs/<run_id>/`. Une mesure non exécutée s'affiche `non exécuté`,
  jamais `0`.
- **Données entièrement synthétiques.** Les rendez-vous et les motifs relèvent potentiellement
  de données de santé : aucune donnée réelle n'entre dans le dépôt, et les traces publiées sont
  pseudonymisées. Voir [`docs/safety.md`](docs/safety.md).
- **Runs non reproductibles écartés.** Un run exécuté avec un dépôt modifié non commité est
  marqué `dirty` et ne remplace jamais les résultats publiés.

## Résultats

<!-- BENCHMARK_RESULTS_START -->

_Aucune campagne publiable n'a encore été exécutée. Cette zone est générée par `ivr-bench readme update` et ne doit pas être éditée à la main._

| Architecture | Tool accuracy | Macro F1 | Rappel urgence | Rappel no_tool | Argument EM | Hallucination | p95 | Cas |
|---|---|---|---|---|---|---|---|---|
| A0 rules | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté |
| A1 diet | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté |
| A2 needle_full | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté |
| A3 functiongemma_zero_shot | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté |
| A4 functiongemma_tuned | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté |
| A5 embedding_only | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté |
| A6 hybrid_needle_top2 | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté |
| A7 hybrid_functiongemma_top2 | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté |
| A8 hybrid_adaptive | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté |

Une cellule `non exécuté` signifie exactement cela : la mesure n'a pas été faite. Elle ne vaut pas zéro.

<!-- BENCHMARK_RESULTS_END -->

## Documentation

| Document | Contenu |
|---|---|
| [`docs/SPECIFICATION_IVR_ROUTING_BENCHMARK.md`](docs/SPECIFICATION_IVR_ROUTING_BENCHMARK.md) | cahier des charges complet |
| [`docs/architecture.md`](docs/architecture.md) | découpage des responsabilités |
| [`docs/dataset.md`](docs/dataset.md) | génération, anti-fuite, manifestes |
| [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) | métriques et protocole statistique |
| [`docs/safety.md`](docs/safety.md) | politique de sécurité et données de santé |
| [`docs/reproducibility.md`](docs/reproducibility.md) | traçabilité des runs |
| [`docs/adding_an_architecture.md`](docs/adding_an_architecture.md) | ajouter un routeur |
| [`docs/adding_a_function.md`](docs/adding_a_function.md) | ajouter une fonction métier |
| [`docs/results_interpretation.md`](docs/results_interpretation.md) | lecture des résultats |
| [`docs/references.md`](docs/references.md) | sources et versions consultées |

## Licence

MIT, voir [`LICENSE`](LICENSE).
