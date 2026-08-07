# Lecture des résultats

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

## Cellules `non exécuté`

Elles signifient exactement cela : la mesure n'a pas été faite. Elles ne valent pas zéro et ne
doivent pas être interpolées.

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
