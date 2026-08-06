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

## Composition publiée, pas supposée

Chaque manifeste publie la répartition par fonction, par sous-suite, **et par
origine** — énoncé issu directement d'un gabarit, ou obtenu par déformation.
Un déficit par rapport à la cible de 300 est publié plutôt que comblé
silencieusement.

Limite connue de la version actuelle : les fonctions sans emplacement variable
(informations générales, transferts, hors périmètre) disposent de peu de
gabarits, si bien que la majorité de leur volume provient des déformations
plutôt que de formulations distinctes. Le champ `by_origin` du manifeste rend ce
déséquilibre visible ; l'enrichir demande d'écrire davantage de gabarits, pas de
changer le moteur.

## Ce que la déduplication fait, et ne fait pas

Un plafond par **noyau sémantique** empêche la même question redite avec une
autre formule de politesse de remplir l'index. Ce plafond ne s'applique
volontairement pas aux déformations : une hésitation ou une erreur de
transcription conserve le sens, et c'est précisément ce qu'on veut mesurer.
