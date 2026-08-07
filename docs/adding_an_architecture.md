# Ajouter une architecture

1. Créer une classe implémentant le protocole `Router` (§9) et renvoyant une
   `RouterPrediction`.
2. L'enregistrer sous un nom stable avec `register("mon_architecture")`, puis importer le
   module dans `src/ivr_bench/routers/__init__.py` — un routeur non importé reste invisible
   du registre.
3. Ajouter la ligne correspondante dans `ARCHITECTURES`
   (`src/ivr_bench/reporting/readme.py`) pour qu'elle apparaisse au tableau, `non exécuté`
   tant qu'aucune campagne ne l'a mesurée.
4. Lancer la suite contractuelle : tous les routeurs passent exactement les mêmes tests.

Aucun autre fichier ne doit être modifié. Un routeur ne produit jamais d'identifiant de
praticien et n'invente jamais d'argument absent de la demande.

## Options de campagne

Les réglages du fichier de campagne (`config/benchmark/cpu.yaml`) sont proposés à toutes les
architectures ; le registre ne transmet à chacune que ceux que la signature de son
constructeur accepte. Une architecture qui n'a pas d'index n'a donc rien à déclarer, et
l'appelant n'a pas à savoir qui accepte quoi.

## Analyse en lot

Une architecture dont le modèle coûte plusieurs minutes à charger peut exposer
`prepare(utterances: list[str]) -> None`. Le harnais l'appelle **avant** le chronomètre :
le chargement appartient au coût de démarrage, pas à la latence d'inférence. C'est ce que
fait le sidecar DIET, qui analyse tout le lot en une passe et rapporte pour chaque énoncé
la latence mesurée au plus près du modèle.

Un routeur sans `prepare()` n'est pas concerné : le harnais ne l'appelle que s'il existe.
