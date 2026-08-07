# NLU-EXPERIMENTAL-ARCH

Banc d'essai reproductible pour le routage d'intentions et l'appel de fonctions dans un
serveur vocal interactif francophone.

Le dépôt compare des stratégies capables de transformer **une phrase** en une fonction métier
structurée, dans le cadre d'une plateforme de prise de rendez-vous médicaux. Ce n'est **pas**
un assistant médical : aucune question clinique n'est traitée, elle est refusée ou transférée
selon une politique déterministe implémentée hors modèle.

Le périmètre s'arrête à la phrase. La reconnaissance vocale se situe en amont : elle ajoute une
source de bruit qui masquerait ce que la comparaison cherche à mesurer, et les suites audio
décrites aux §14–15 de la spécification sont donc hors périmètre. L'entrée est du texte.

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
| A9 | Classifieur sur embeddings | régression logistique, EmbeddingGemma |
| A10 | Classifieur lexical | TF-IDF caractères, aucun réseau |
| A11 | k plus proches voisins | vote pondéré sur l'index |
| A12 | Questions hypothétiques, delta cosinus | proposition d'origine, top 2 puis Needle |
| A13 | Classifieur puis DIET | la fonction vient d'A9, les arguments de DIET |
| A14 | Classifieur, DIET, puis règles | idem, arguments manquants complétés par règles |
| A15 | Arbitrage par argument | source choisie argument par argument sur la validation |
| A16 | Arbitrage et énumérations apprises | idem, plus un classifieur par argument énuméré |
| A17 | Classifieur et énumérations apprises | contrôle : A16 sans DIET |

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

_Dernière campagne le 2026-08-07T11:13:22+00:00 — x86_64, 4 fils, 15.7 Go, accélérateur : aucun (comparaison CPU par construction). Les 16 lignes proviennent de campagnes distinctes, réparties sur 9 commits ; chacune est tracée dans `results/runs/`._

| Architecture | Appel exact | Tool accuracy | Macro F1 | Rappel urgence | Rappel no_tool | Argument EM | Hallucination | p95 | Cas |
|---|---|---|---|---|---|---|---|---|---|
| A0 rules | 46.3% | 62.4% | 63.7% | 35.3% | 100.0% | 84.4% | 0.0% | 0 ms | 2088 |
| A1 diet | 27.4% | 65.9% | 62.0% | 80.7% | 14.7% | 76.3% | 0.6% | 18 ms | 2088 |
| A2 needle_full | 0.0% | 11.9% | 4.1% | 0.0% | 0.0% | 58.0% | 8.0% | 8841 ms | 84 / 2088 |
| A3 functiongemma_zero_shot | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | non exécuté | non exécuté | 2812 ms | 84 / 2088 |
| A4 functiongemma_tuned | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté | non exécuté |
| A5 embedding_only | 30.3% | 68.1% | 64.6% | 95.3% | 10.7% | 74.0% | 0.0% | 989 ms | 2088 |
| A6 hybrid_needle_top2 | 3.6% | 20.2% | 19.5% | 0.0% | 25.0% | 53.0% | 15.2% | 9088 ms | 84 / 2088 |
| A7 hybrid_functiongemma_top2 | 1.2% | 19.0% | 22.5% | 0.0% | 0.0% | 18.1% | 47.2% | 2926 ms | 84 / 2088 |
| A8 hybrid_adaptive | 3.6% | 19.0% | 16.0% | 0.0% | 25.0% | 50.8% | 14.3% | 5834 ms | 84 / 2088 |
| A9 embedding_classifier | 34.0% | 75.6% | 74.6% | 99.3% | 44.0% | 73.7% | 0.0% | 137 ms | 2088 |
| A10 lexical_classifier | 21.6% | 50.4% | 44.7% | 24.7% | 5.0% | 77.8% | 0.0% | 1 ms | 2088 |
| A11 nearest_neighbour | 33.4% | 74.9% | 72.4% | 90.7% | 20.0% | 74.1% | 0.0% | 90 ms | 2088 |
| A12 hypothetical_delta_top2 | 3.6% | 19.0% | 18.4% | 0.0% | 25.0% | 55.7% | 13.1% | 8899 ms | 84 / 2088 |
| A13 classifier_diet | 29.7% | 75.6% | 74.6% | 99.3% | 44.0% | 72.8% | 0.8% | 146 ms | 2088 |
| A14 classifier_diet_rules | 31.8% | 75.6% | 74.6% | 99.3% | 44.0% | 75.3% | 0.8% | 157 ms | 2088 |
| A15 classifier_diet_arbitrated | 32.2% | 75.6% | 74.6% | 99.3% | 44.0% | 75.5% | 0.8% | 156 ms | 2088 |
| A16 classifier_diet_enum | 40.1% | 75.6% | 74.6% | 99.3% | 44.0% | 79.7% | 0.8% | 148 ms | 2088 |

Une cellule `non exécuté` signifie exactement cela : la mesure n'a pas été faite. Elle ne vaut pas zéro.

**Appel exact** : la fonction et *tous* ses arguments sont corrects, compté sur l'ensemble du corpus. C'est ce qu'un serveur vocal peut exécuter sans reposer de question. **Argument EM** se compte clé par clé et seulement sur les cas où la fonction est correcte, donc sur un sous-ensemble différent pour chaque architecture : les deux colonnes ne classent pas dans le même ordre.

Couverture partielle sur : functiongemma_zero_shot, hybrid_adaptive, hybrid_functiongemma_top2, hybrid_needle_top2, hypothetical_delta_top2, needle_full. Ces architectures coûtent plusieurs secondes par énoncé sur CPU ; l'échantillon est stratifié par fonction et sa taille figure dans la colonne « Cas ».

Résultats bruts : `results/runs/`. Reproduction : `make reproduce`.

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
