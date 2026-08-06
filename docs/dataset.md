# Jeux de données

Tout est synthétique. Aucune donnée réelle, aucune voix réelle, aucun identifiant réel.

## Deux générateurs indépendants

La prévention des fuites (§10.3) ne repose pas sur un filtrage a posteriori mais sur la
construction : `generator_a` produit l'index, l'entraînement et la validation ; `generator_b`
produit le test, avec des gabarits, des banques lexicales et des graines disjoints. La
déduplication lexicale puis sémantique n'intervient qu'en second rideau, et le nombre
d'éléments supprimés est publié dans un rapport de contamination.

## Volumes par fonction

| Partition | Formulations |
|---|---|
| index | 300 positives |
| contrastives | 100 |
| validation | 100 |
| test | 300, protocole de génération distinct |

## Praticiens

Le catalogue synthétique couvre noms français fréquents, noms internationaux, accents,
apostrophes, noms composés, homophones et paires proches. **La partition de test contient des
praticiens absents de l'entraînement** : le routeur doit extraire un nom prononcé, pas
mémoriser une liste fermée.

## État d'implémentation

Générateurs à venir au jalon M3.
