# Protocole de mesure

## Modes

1. **texte oracle** — transcription de référence, mesure le routeur seul ;
2. **transcription ASR** — mesure la dégradation introduite par la reconnaissance vocale ;
3. **bout en bout** — de l'audio à l'action métier.

La suite principale est single-turn. Le remplissage de slots sur plusieurs tours est mesuré
dans une suite secondaire distincte, car les modèles comparés n'ont pas tous vocation à gérer
un dialogue multi-tours.

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
