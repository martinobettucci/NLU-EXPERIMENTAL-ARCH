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

## CPU uniquement

Les colonnes GPU et VRAM du §17.5 sont **non applicables** : la comparaison porte sur des
architectures CPU. C'est un choix expérimental, pas une mesure manquante.

## État d'implémentation

Harnais et métriques à venir au jalon M6.
