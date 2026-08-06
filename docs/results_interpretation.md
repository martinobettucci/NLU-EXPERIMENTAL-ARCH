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

## État d'implémentation

Rapports à venir au jalon M6.
