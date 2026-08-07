# Tableau final : toutes les stratégies, tous les jeux

## Ce que le tableau doit afficher

Une ligne par **stratégie × jeu de données**. Rien de moins que ces colonnes :

| Colonne | Définition | Source |
|---|---|---|
| Stratégie | A0 … A17 | registre |
| Jeu | maison, SNIPS, CLINC150, ATIS, MASSIVE | chargeur |
| **Temps par inférence** | p50 et p95 **par énoncé** | `latency_ms` |
| **Exactitude intention** | fonction correcte | `tool_accuracy` |
| **Exactitude entités** | arguments corrects, clé par clé, sur les cas où la fonction est correcte | `argument_exact_match` |
| **Appel exact** | fonction **et** tous ses arguments, sur tout le corpus | `exact_call_rate` |
| **Hallucination** | argument rempli que l'énoncé n'exprime pas | `hallucinated_argument_rate` |
| **Refus d'appel** | rappel sur les énoncés hors périmètre | `no_tool_recall` / OOS |
| **Temps d'entraînement** | secondes, mesurées, pas estimées | manifeste d'entraînement |
| Cas | effectif réellement évalué / disponible | `coverage` |

Deux colonnes n'existent pas encore et sont à construire : le temps
d'entraînement, et la ligne « jeu » puisque seul le corpus maison tourne.

## Temps d'entraînement : ce qu'on mesure exactement

Trois natures différentes, qui ne se comparent pas si on les additionne sans
le dire :

| Nature | Architectures | Ce qui est chronométré |
|---|---|---|
| Aucun entraînement | A0, A2, A3, A5, A6, A7, A8, A12 | zéro, déclaré tel quel |
| Ajustement en mémoire | A9, A10, A11, A13 – A17 | construction du routeur, dans le harnais |
| Entraînement hors bande | A1 DIET, A4 FunctionGemma LoRA | commande dédiée, manifeste écrit à la fin |

Un ajustement de régression logistique et un entraînement DIET de sept minutes
ne sont pas la même dépense : la colonne porte les secondes, la nature figure
en note. Les architectures hybrides paient l'entraînement de leurs deux
composants — c'est un coût réel de la composition, il est compté.

Le coût de **démarrage** est une autre colonne, déjà mesurée (`warmup_ms`) : ce
qu'il faut payer à chaque redémarrage du processus, indépendamment de
l'entraînement.

## Ce que chaque jeu teste

| Jeu | Rôle | Intentions | Slots | Hors périmètre |
|---|---|---|---|---|
| maison (fr) | cas d'usage réel, 7 fonctions | 7 | oui | oui (`no_tool`) |
| SNIPS | contrôle — l'architecture classe-t-elle l'évident ? | 7 | oui | non |
| ATIS | confusion sémantique : `flight` contre `airfare` sur le même vocabulaire | ~26 | oui | non |
| CLINC150 | **le plus informatif** : 150 fonctions proches + rejet explicite | 150 | non | oui |
| MASSIVE | passage à l'échelle, énoncés parallèles | 60 | oui | non |

CLINC150 est celui qui met réellement l'hypothèse à l'épreuve : 150 fonctions
dont beaucoup se recouvrent (`cash_withdrawal_charge` contre
`cash_withdrawal_wrong_exchange_rate`), plus un jeu hors périmètre. C'est
exactement le régime où la préselection sémantique devrait payer — et où un
plus-proche-voisin naïf répond toujours quelque chose.

Sur SNIPS, ATIS et MASSIVE, la colonne « refus d'appel » vaut **non
applicable** : ces jeux ne contiennent aucun énoncé hors périmètre. Elle ne
vaut pas zéro.

## Le point qui n'est pas cosmétique

Ces jeux portent des **intentions**, notre banc porte des **fonctions
appelables**. La conversion n'est pas un détail d'ingénierie : c'est elle qui
décide si la comparaison a un sens.

Chaque intention devient une définition de fonction, avec une description
rédigée à partir de ses exemples d'entraînement, puis le générateur de
questions hypothétiques travaille **sur cette définition** — jamais sur les
énoncés de test. Les slots du jeu deviennent les arguments de la fonction.
Sans cette règle, l'architecture verrait le test pendant sa construction et
tous les chiffres seraient faux.

## Échantillonnage, et pourquoi il est déclaré

**500 énoncés d'évaluation par jeu**, stratifiés par intention, graine publiée.

Ce chiffre change la nature de l'exercice, et il faut le dire : à 500 cas,
l'intervalle de confiance à 95 % d'une exactitude de 75 % fait environ
±3,8 points. Deux architectures séparées de trois points ne seront pas
distinguables, et le test apparié le dira explicitement plutôt que de laisser
lire un classement dans le tableau. C'est un tableau d'ordres de grandeur, pas
de départages fins — et c'est suffisant pour la question posée : une stratégie
tient-elle hors de nos propres gabarits ?

Une architecture à cinq secondes par énoncé ne tient pas non plus 500 cas :
ce serait quarante minutes pour **une seule case**, seize heures pour la
colonne entière des micro-modèles. Ces cellules gardent un échantillon
stratifié plus petit, et **la colonne « Cas » porte l'effectif réel de chaque
cellule**. Une cellule à 84 cas et une cellule à 500 ne se comparent pas sans
le savoir ; le tableau le dit plutôt que de le laisser deviner.

Les comparaisons appariées (McNemar) ne portent que sur les cas partagés.

**L'entraînement, lui, n'est pas échantillonné.** Réduire le corpus
d'entraînement changerait ce qu'on mesure : une architecture apprend sur le
train complet du jeu, comme le font les travaux auxquels ces chiffres se
compareront. Seule l'évaluation est échantillonnée.

## Sources vérifiées

Identifiants et formats constatés, pas supposés — chaque dépôt a été interrogé
avant d'écrire un chargeur.

| Jeu | Dépôt | Format | Slots |
|---|---|---|---|
| SNIPS | `bkonkle/snips-joint-intent` | `train.csv` / `test.csv`, colonnes `input,intent,slots` | oui, BIO |
| ATIS | `tuetschek/atis` | `atis_train.csv` / `atis_test.csv`, colonnes `id,intent,text,slots` | oui, BIO |
| CLINC150 | `clinc/clinc_oos`, variante `plus` | parquet, train/validation/test | non, intentions seules |
| MASSIVE | `AmazonScience/massive`, config `fr-FR` | parquet converti (le script de chargement n'est plus exécutable sous `datasets` 5) | oui |

Les slots BIO se convertissent en arguments par regroupement des segments
`B-`/`I-` : un slot devient un argument, sa valeur est le texte du segment.
C'est la même opération que l'annotation du corpus maison pour DIET, en sens
inverse.

## Étapes

- [ ] **P0 — Temps d'entraînement.** Chronométrage de la construction du
      routeur dans le harnais (`build_seconds`), manifeste écrit par les
      commandes d'entraînement hors bande (`results/training/<arch>.json`),
      colonne dans le tableau. Sans jeu public, cette étape est déjà utile au
      tableau actuel.
- [ ] **P1 — Chargeurs.** Un adaptateur par jeu vers `GeneratedCase`, avec
      téléchargement explicite (`ivr-bench datasets download`) et manifeste
      d'empreintes. Aucun téléchargement implicite pendant les tests.
- [ ] **P2 — Échantillonnage.** Stratifié par intention, 500 cas d'évaluation
      par jeu, graine publiée, effectif réel déclaré par cellule. Train complet.
- [ ] **P3 — Conversion intention → fonction.** Description générée depuis les
      exemples d'entraînement seuls ; slots → arguments ; vérification qu'aucun
      énoncé de test n'a servi à construire une définition.
- [ ] **P4 — Génération hypothétique par jeu.** Le même générateur que le
      corpus maison, appliqué aux définitions converties.
- [ ] **P5 — Index et vecteurs par jeu.** Un index sémantique par jeu, et
      surtout le cache des vecteurs du train : CLINC150 compte 15 000 énoncés
      d'entraînement, soit dix-huit minutes d'encodage. Les réencoder pour
      chacune des douze architectures apprenantes coûterait quatre heures de
      calcul identique. Encodage une fois par jeu, réutilisé par toutes.
- [ ] **P6 — Campagnes.** Toutes les architectures × tous les jeux, séquencées
      sous verrou, charge machine enregistrée.
- [ ] **P7 — Tableau final.** Une ligne par stratégie × jeu, avec toutes les
      colonnes ci-dessus. Intervalles bootstrap et McNemar apparié sur les cas
      partagés.

## Décisions tranchées

Elles étaient ouvertes ; les laisser ouvertes bloquerait P1. Chacune est
déclarée dans le manifeste du jeu, et réversible par configuration.

1. **MASSIVE : `fr-FR` uniquement.** Le corpus maison est francophone ;
   comparer à périmètre égal est la question posée. L'ajout de `en-US` et
   `it-IT` testerait la promesse multilingue de l'espace d'embedding partagé —
   c'est une autre expérience, laissée en ablation §27.
2. **CLINC150 : variante `plus`.** C'est celle qui porte le plus d'énoncés hors
   périmètre, donc celle qui rend la colonne « refus d'appel » informative.
   Les taux publiés ailleurs ne sont comparables qu'à variante égale, et la
   variante est écrite dans le manifeste.
3. **ATIS : intentions composées conservées.** Aucun prétraitement, aucune
   fusion de `flight+airfare` : le nombre de classes réellement observé est
   compté et publié. Fusionner améliorerait mécaniquement les scores sans
   qu'aucune architecture n'ait progressé.

## Ce que ce tableau ne dira pas

Les latences resteront celles d'un CPU 4 cœurs sans accélérateur. Elles
classent les architectures entre elles ; elles ne se comparent pas aux chiffres
publiés par d'autres sur d'autres machines.
