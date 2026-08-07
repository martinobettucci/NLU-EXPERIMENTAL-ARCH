# Protocole de mesure

## Périmètre : la phrase, pas le signal

La question mesurée est : *à partir d'une phrase, quelles stratégies retrouvent la bonne
fonction et ses arguments ?* Les modes audio des §14–15 sont écartés — la reconnaissance
vocale précède le problème et n'ajoute qu'une source de bruit entre les architectures
comparées. Le seul mode exécuté est donc le mode texte.

Cette restriction est un choix de périmètre, pas une mesure manquante : les colonnes
correspondantes sont marquées non applicables plutôt que `non exécuté`.

La suite est single-turn : une phrase, une fonction. C'est exactement la question posée.

## Classement

L'ordre de lecture des résultats est imposé (§18) : réussite métier bout en bout, exactitude
de la fonction, rappel de sécurité, validité des arguments, latence p95, mémoire maximale.
Aucun score composite unique ne masque ces compromis ; une vue de Pareto est produite.

## Statistiques

Cinq graines pour les composants entraînés, intervalles de confiance bootstrap à 95 %,
comparaison appariée et test de McNemar. Le nombre exact d'exemples, les exclusions et les
erreurs sont publiés. Aucun agrégat n'est publié sans résultat par catégorie.

## Ce que « latence » désigne exactement

Une seule chose : **le temps d'un énoncé**, du texte reçu à l'appel de fonction produit.

| Où | Quoi |
|---|---|
| `latency_ms` dans `predictions.jsonl` | un énoncé, chronomètre autour de `predict()` — récupération, décodage et validation comprises |
| `decode_ms`, `encode_ms`, `search_ms` dans `metadata` | décomposition du même énoncé, étape par étape |
| `latency_ms.p50 / p95 / p99 / mean` dans `metrics.json` | percentiles **sur ces valeurs par énoncé**, jamais sur un total divisé |
| colonne `p95` du README | le p95 par énoncé |
| `duration_s` dans `environment.json` | durée totale de la boucle de mesure, à titre indicatif |
| `warmup_ms` dans `metrics.json` | coût de démarrage, mesuré séparément (voir ci-dessous) |

Pour DIET, la latence vient du sidecar, relevée au plus près du modèle : elle exclut le
transport par fichier, qui n'appartient pas à l'architecture.

## Chargement des poids : une fois, et hors mesure

Chaque runtime charge son modèle **une seule fois par processus** (`lru_cache` sur le
chargeur). Mais les poids se chargent à la première inférence, pas à la construction du
routeur : sans précaution, le premier énoncé porte le chargement et la compilation.

Ce n'était pas une précaution théorique. Avant correction : 22 036 ms pour le premier appel de
Needle contre 4 772 ms de médiane, 11 562 ms pour le premier appel du classifieur contre 95 ms.
Le p50 et le p95 n'en étaient pas affectés — un point isolé n'atteint pas le 95ᵉ centile — mais
la moyenne et le p99 l'étaient entièrement.

Le harnais exécute donc une **passe de chauffe** sur le premier cas réel, hors chronomètre, et
publie son coût dans `warmup_ms`. Ce coût n'est pas effacé : c'est lui qui décide si une
architecture est déployable sur une machine qui redémarre souvent.

Les campagnes antérieures à cette correction gardent leur `warmup_ms` à `null` : leur moyenne
et leur p99 incluent le démarrage, leur p50 et leur p95 sont comparables.

## CPU uniquement

Les colonnes GPU et VRAM du §17.5 sont **non applicables** : la comparaison porte sur des
architectures CPU. C'est un choix expérimental, pas une mesure manquante.

## Reproduire

`make reproduce` rejoue les campagnes publiées. Chaque run enregistre sa commande exacte,
son commit, les empreintes de son corpus et la charge machine relevée avant et après la
mesure. Les campagnes sont sérialisées par un verrou exclusif : deux campagnes simultanées ne
mesureraient que la contention.
