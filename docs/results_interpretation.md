# Lecture des résultats

## Deux colonnes, deux classements

**Appel exact** compte les énoncés dont la fonction *et* tous les arguments sont corrects, sur
l'ensemble du corpus. **Argument EM** compte clé par clé, et seulement sur les cas où la
fonction est correcte — donc sur un sous-ensemble différent pour chaque architecture.

Les deux ne classent pas pareil, et l'écart est instructif : A14 gagne 1,6 point d'Argument EM
sur A9 et lui perd 2,2 points d'appel exact. Une moyenne par clé absorbe une erreur qu'un
appelant subit en entier. C'est aussi pourquoi les 84,4 % d'Argument EM de la baseline de
règles ne signifient pas qu'elle extrait mieux : ils portent sur les 62,4 % de cas les plus
simples, ceux qu'elle est la seule à ne pas manquer.

## Séparer le choix de la fonction et l'extraction des arguments

A13 à A16 confient la fonction au classifieur d'A9 et les arguments à DIET. Comme le
classifieur est identique, ces architectures prédisent **exactement les mêmes fonctions** que
A9 : l'écart mesuré porte sur l'extraction et sur rien d'autre. C'est la seule comparaison
d'extracteurs du dépôt qui soit appariée.

Argument par argument, sur les cas où la fonction est correcte :

| argument | n | règles | DIET |
|---|---|---|---|
| practitioner_name | 716 | 75,4 % | **93,3 %** |
| preferred_time | 292 | 69,2 % | **77,1 %** |
| preferred_new_time | 269 | 21,6 % | **31,2 %** |
| preferred_date | 292 | **81,8 %** | 43,8 % |
| preferred_new_date | 269 | **94,8 %** | 74,3 % |
| date_from | 155 | **88,4 %** | 68,4 % |
| specialty | 292 | **100 %** | 96,9 % |

DIET lit les noms propres nettement mieux que les règles, et les règles lisent les dates
nettement mieux que DIET. Prendre une source en bloc — c'est ce que font A13 et A14 — laisse
donc de la précision des deux côtés, et aucune des deux ne dépasse le classifieur seul sur
l'appel exact.

## L'arbitrage transfère mal, et c'est le dispositif anti-fuite qui le dit

A15 choisit la source argument par argument, d'après ce qu'il mesure sur la **validation**.
Le gain est réel mais faible (32,2 % contre 31,8 % pour A14), parce que la validation ne dit
pas la même chose que le test : sur `preferred_date` l'arbitrage retient DIET, alors que le
test donne les règles gagnantes de 38 points.

Ce n'est pas un défaut de l'arbitrage, c'est le §10.3 qui produit son effet. Validation et test
viennent de deux familles de gabarits disjointes : ce qu'on calibre sur l'une ne se transporte
pas automatiquement sur l'autre. Un dépôt qui aurait tiré ses deux jeux du même générateur
aurait vu l'arbitrage « marcher » — et aurait mesuré sa propre fuite.

## Le plus gros levier n'était pas dans l'hybridation

`topic` compte quatorze valeurs possibles et environ trois cents cas. L'extracteur partagé par
A0, A5, A9, A10 et A11 y répond toujours `other`, et DIET ne peut pas mieux faire : un argument
énuméré n'apparaît pas littéralement dans la phrase, il n'y a donc aucun segment à annoter.
Résultat : **0 %** pour tout le monde, sur 11 % du corpus.

A16 entraîne un petit classifieur lexical par argument énuméré, sur `train`, le même corpus que
tout le monde. Le résultat est le plus grand écart mesuré dans ce dépôt :

| argument | A15 | A16 |
|---|---|---|
| reason | 86,7 % | 99,2 % |
| reason_category | 33,2 % | 48,0 % |
| topic | 0 % | 26,1 % |

L'appel exact passe de 32,2 % à **40,1 %**. Un argument dont la valeur ne figure pas dans la
phrase relève d'une classification, pas d'une extraction : aucune combinaison de deux
extracteurs ne pouvait le trouver.

## Le contrôle qui retire DIET

A16 change trois choses à la fois par rapport au classifieur seul — les entités de DIET,
l'arbitrage, les énumérations apprises — et son avance ne dit pas laquelle les a gagnées.
A17 ne garde que la dernière : même classifieur, même extracteur par règles, aucune trace de
DIET.

| | appel exact | Argument EM | Hallucination | p95 |
|---|---|---|---|---|
| A17 sans DIET | **41,9 %** | 77,8 % | **0,0 %** | **136 ms** |
| A16 avec DIET | 40,1 % | **79,7 %** | 0,8 % | 147 ms |
| A9 référence | 34,0 % | 73,7 % | 0,0 % | 137 ms |

**Retirer DIET améliore le résultat.** L'écart argument par argument dit pourquoi : DIET gagne
17,9 points sur `practitioner_name` et 8 à 10 points sur les heures, mais perd 11 à 15 points
sur chacune des quatre dates. Un appel exige *tous* ses arguments : les dates perdues coûtent
plus d'appels complets que les noms gagnés n'en rapportent. DIET introduit en prime 0,8 %
d'hallucination là où les règles n'en produisent aucune, et 10 ms de latence.

C'est la deuxième fois que l'Argument EM classe à l'envers de l'appel exact, et cette fois au
sommet du tableau. Une moyenne par clé récompense un extracteur qui a souvent un peu raison ;
un serveur vocal a besoin d'un extracteur qui a entièrement raison.

**Réponse à la question posée** : hybrider le classifieur d'intentions avec DIET pour les
entités n'améliore pas ce banc d'essai. Ce qui l'améliore — de 34,0 % à 41,9 %, soit près de
huit points — c'est d'avoir vu que `topic`, `reason` et `reason_category` posaient une question
de classification déguisée en question d'extraction.

Il reste une raison de garder DIET à l'esprit : il est le seul à lire correctement les noms
propres (93,3 % contre 75,4 %). Un extracteur de dates meilleur du côté DIET, ou un arbitrage
par argument qui transfère, replacerait la question. En l'état, la mesure dit non.

## Ce que le tableau principal ne dit pas

Une exactitude de fonction élevée ne suffit pas. Un système qui choisit la bonne fonction mais
invente un argument absent de la demande est plus dangereux qu'un système qui laisse le champ
vide et pose une question. Les métriques pénalisent donc plus fortement un argument halluciné
qu'un argument manquant.

De même, une résolution de praticien erronée avec forte confiance est plus grave qu'une demande
de clarification.

## Latences

Les latences Needle sont mesurées avec le runtime JAX sur CPU, pas avec le runtime natif de
Cactus, qui n'est pas distribué pour cette plateforme. Le chiffre publié est donc un plancher
pessimiste et n'est pas comparable aux débits annoncés par l'auteur du modèle.

Les lignes du tableau viennent de campagnes lancées à des heures différentes, et la machine
n'est pas également chargée d'une heure à l'autre : le même classifieur a mesuré 64 ms puis
95 ms de médiane à quelques heures d'écart, sans qu'une ligne de son code ait changé. Chaque
run enregistre la charge relevée avant et après (`machine_load` dans `metrics.json`), et les
campagnes sont sérialisées par un verrou exclusif. Les comparaisons de latence n'ont de sens
qu'entre architectures mesurées à la suite : c'est le cas d'A9, A13, A14, A15 et A16, lancées
dans cet ordre sans rien d'autre sur la machine.

## Cellules `non exécuté`

Elles signifient exactement cela : la mesure n'a pas été faite. Elles ne valent pas zéro et ne
doivent pas être interpolées.

## La stratégie la plus efficace n'est pas celle qu'on comparait

Une régression logistique entraînée sur les mêmes embeddings que le retriever
atteint **75,6 %** d'exactitude et **99,3 %** de rappel d'urgence, en 75 ms au
p95. Elle dépasse le retriever seul (68,1 %), la baseline de règles (62,4 %) et
toutes les architectures adossées à un micro-modèle d'appel d'outils, qui
plafonnent à 20,2 % pour cinquante fois plus de latence.

La différence tient à la question posée. Le retriever demande « de quoi cette
phrase est-elle proche ? » et compare à des prototypes ; le classifieur apprend
« qu'est-ce qui sépare ces fonctions ? ». Sur un catalogue fermé de sept
fonctions, la seconde question est la bonne — et elle ne nécessite aucun modèle
génératif en aval.

Le classifieur lexical TF-IDF, sans aucun réseau de neurones, atteint 50,4 % en
1 ms. C'est nettement moins, mais cela chiffre ce que l'encodeur dense apporte
réellement : environ 25 points, pour un modèle de 300 millions de paramètres.

## Ce que disent les premières campagnes

Aucune architecture adossée à Needle ne sélectionne jamais `emergency_handoff` :
son rappel est de 0 % sur les trois variantes. C'est la métrique de sécurité
prioritaire du §17.1, et ce résultat suffit à disqualifier ces configurations
pour un usage réel, quelle que soit leur exactitude par ailleurs.

La préselection sémantique fait passer Needle de 11,9 % à 20,2 % : réduire le
catalogue à deux candidats aide nettement. Mais le point de départ est si bas
que l'écart ne tranche pas encore l'hypothèse du dépôt — il faudra FunctionGemma
pour savoir si la préselection aide un modèle qui, lui, maîtrise le domaine.

## La préselection aide, et c'est mesuré

Sur les mêmes 84 cas appariés, réduire le catalogue à deux candidats fait
passer :

- Needle de 11,9 % [6,0 – 19,0] à 20,2 % [11,9 – 28,6] — McNemar, p = 0,016 ;
- FunctionGemma de 0,0 % à 19,0 % [10,7 – 27,4] — McNemar, p < 0,001.

C'est l'hypothèse centrale du dépôt, et elle tient sur les deux modèles d'appel
d'outils, pas sur un seul.

## Ce que cela ne dit pas

Les deux architectures hybrides restent très loin du retriever seul (68,1 % sur
l'ensemble du corpus). Le gain est donc réel mais part d'un plancher : aucun des
deux micro-modèles ne maîtrise ce domaine francophone en l'état. FunctionGemma
zéro-shot n'appelle correctement aucune fonction — il répond en prose anglaise —
ce qui correspond à ce que son éditeur annonce d'une base destinée à être
spécialisée. La question « la préselection suffit-elle à rendre un micro-modèle
utilisable ici ? » reste donc ouverte, et c'est la variante spécialisée (A4) qui
peut y répondre.

Les intervalles sont larges parce que l'échantillon est de 84 cas. Deux
architectures séparées d'un point ne sont pas distinguables : le test apparié le
dit explicitement plutôt que de laisser lire un classement dans le tableau.

## Question ouverte

Le taux de sorties invalides **augmente** quand on réduit le nombre d'outils
offerts : 8 cas sur 84 avec sept fonctions, 29 sur 84 avec deux. C'est
contre-intuitif. Les cas rejoués isolément avec la bonne fonction dans le lot
redeviennent valides, ce qui oriente vers des arguments empruntés au schéma
d'une autre fonction — mais le mécanisme n'a pas été confirmé, et il n'est donc
pas présenté comme établi.
