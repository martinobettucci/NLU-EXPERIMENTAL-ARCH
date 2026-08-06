# Reproductibilité

Chaque run enregistre son contexte complet dans `results/runs/<run_id>/environment.yaml` :
identifiant, horodatage, commit, état propre ou modifié, système, noyau, version de Python,
dépendances, empreinte du conteneur, processeur, threads, mémoire, runtime, modèles,
quantification, graines, empreintes des jeux de données, empreinte de configuration, commande
exacte et durée.

## Runs modifiés

Un run exécuté avec un dépôt modifié non commité est marqué `dirty`. Il reste consultable mais
ne remplace jamais les résultats publiés dans le README.

## Traçabilité

Chaque chiffre publié doit être remontable jusqu'à un fichier de run. La zone générée du README
est produite par `ivr-bench readme update` et n'est jamais éditée à la main.

## Reproduction

```bash
make reproduce
```

## État d'implémentation

Journalisation des runs à venir au jalon M6.
