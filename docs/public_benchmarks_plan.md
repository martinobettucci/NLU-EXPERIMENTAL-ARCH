# Tableau final : jeux publics + corpus maison

## Objectif

Un seul tableau comparant **toutes** les architectures sur des chiffres
comparables à ceux que d'autres publient. Le corpus maison mesure le cas
d'usage réel ; les jeux publics disent si l'architecture tient hors de nos
propres gabarits.

Échantillon : **5 000 énoncés maximum par jeu**, stratifiés par intention,
tirage seedé et publié.

## Ce que chaque jeu teste

| Jeu | Rôle | Intentions | Slots | Hors périmètre |
|---|---|---|---|---|
| maison (fr) | cas d'usage réel, 7 fonctions | 7 | oui | oui (`no_tool`) |
| SNIPS | contrôle — l'architecture classe-t-elle l'évident ? | 7 | oui | non |
| ATIS | confusion sémantique : `flight` contre `airfare` sur le même vocabulaire | ~26 | oui | non |
| CLINC150 | **le plus informatif** : 150 fonctions proches + rejet explicite | 150 | non | oui |
| MASSIVE | passage à l'échelle et multilingue, énoncés parallèles | 60 | oui | non |

CLINC150 est celui qui met réellement l'hypothèse à l'épreuve : 150 fonctions
dont beaucoup se recouvrent (`cash_withdrawal_charge` contre
`cash_withdrawal_wrong_exchange_rate`), plus un jeu hors périmètre. C'est
exactement le régime où la préselection sémantique devrait payer — et où un
plus-proche-voisin naïf répond toujours quelque chose.

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

## Étapes

- [ ] **P1 — Chargeurs.** Un adaptateur par jeu vers `GeneratedCase`, avec
      téléchargement explicite (`ivr-bench datasets download`) et manifeste
      d'empreintes. Aucun téléchargement implicite pendant les tests.
- [ ] **P2 — Échantillonnage.** Stratifié par intention, 5 000 max, graine
      publiée, effectif réel déclaré par jeu.
- [ ] **P3 — Conversion intention → fonction.** Description générée depuis les
      exemples d'entraînement seuls ; slots → arguments ; vérification qu'aucun
      énoncé de test n'a servi à construire une définition.
- [ ] **P4 — Génération hypothétique par jeu.** Le même générateur que le
      corpus maison, appliqué aux définitions converties.
- [ ] **P5 — Index par jeu.** Un index sémantique par jeu et par dimension
      d'embedding.
- [ ] **P6 — Campagnes.** Toutes les architectures × tous les jeux, séquencées
      (verrou de campagne), avec la charge machine enregistrée.
- [ ] **P7 — Tableau final.** Une ligne par architecture, une colonne par jeu :
      exactitude, macro F1, rejet hors périmètre, extraction d'arguments,
      p50/p95, mémoire, coût d'entraînement. Intervalles bootstrap et McNemar
      apparié sur les cas partagés.

## Décisions à trancher avant P1

1. **MASSIVE, quelles langues ?** Le corpus maison est francophone. Se limiter
   à `fr-FR` compare à périmètre égal ; ajouter `en-US` et `it-IT` teste la
   promesse multilingue de l'espace d'embedding partagé. Les deux sont
   défendables, ce n'est pas la même expérience.
2. **CLINC150, quelle variante ?** `small`, `imbalanced`, `plus` ou `full` —
   les taux de rejet publiés ne sont pas comparables entre variantes.
3. **ATIS avec ou sans les intentions composées** (`flight+airfare`) : les
   effectifs publiés vont de 17 à 26 classes selon le prétraitement, et le
   chiffre retenu doit être déclaré.

## Ce que ce tableau ne dira pas

Les latences resteront celles d'un CPU 4 cœurs sans accélérateur. Elles
classent les architectures entre elles ; elles ne se comparent pas aux chiffres
publiés par d'autres sur d'autres machines.
