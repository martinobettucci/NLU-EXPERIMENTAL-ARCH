# Ajouter une architecture

1. Créer une classe implémentant le protocole `Router` (§9) et renvoyant une
   `RouterPrediction`.
2. L'enregistrer dans `src/ivr_bench/routers/registry.py`.
3. Ajouter un fichier de configuration dans `config/architectures/`.
4. Lancer la suite contractuelle : tous les routeurs passent exactement les mêmes tests.

Aucun autre fichier ne doit être modifié. Un routeur ne produit jamais d'identifiant de
praticien et n'invente jamais d'argument absent de la demande.

## État d'implémentation

Registre et suite contractuelle à venir aux jalons M1 et M5.
