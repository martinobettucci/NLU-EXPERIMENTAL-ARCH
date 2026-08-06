# Ajouter une fonction métier

La source canonique est `config/domain/functions.yaml`. Tous les adaptateurs — définitions
d'outils Needle, format FunctionGemma, JSON Schema, domaine Rasa, motifs de règles — en sont
dérivés.

Pour l'architecture hybride :

```text
nouvelle définition → génération offline → embeddings → insertion dans l'index
```

Pour DIET :

```text
nouvelle intention → exemples → réentraînement → nouvelle évaluation complète
```

Cette asymétrie est mesurée explicitement (§26) par l'ajout de
`request_appointment_cancellation` après la première campagne : code modifié, données ajoutées,
temps machine, réentraînement requis, régression sur les fonctions existantes.

## État d'implémentation

Expérience d'ajout à venir au jalon M9.
