# Architecture

## Séparation des responsabilités

Le système distingue quatre sous-problèmes, et le banc d'essai les mesure séparément (§4) :

1. reconnaissance vocale ;
2. sélection de la fonction métier ;
3. extraction des arguments présents dans la demande ;
4. orchestration du dialogue et exécution métier.

Le routeur produit une **fonction atomique**, jamais une chaîne d'appels dépendants. La
résolution du praticien, la recherche de disponibilités, la confirmation et la création
effective sont réalisées par le moteur déterministe en aval. Cette séparation est ce qui rend
comparables des systèmes aussi différents que DIET, Needle et FunctionGemma.

## Contrat commun

Toutes les architectures implémentent le même protocole `Router` et renvoient une
`RouterPrediction` (§9). Un routeur ne produit jamais d'identifiant de praticien : il extrait
le texte prononcé, et le résolveur seul décide de l'identité.

## État d'implémentation

| Composant | Jalon | État |
|---|---|---|
| Contrats et domaine canonique | M1 | à venir |
| Backend synthétique et résolveur | M2 | à venir |
| Générateurs de corpus | M3 | à venir |
| Embeddings et index | M4 | à venir |
| Routeurs A0, A2, A5, A6, A8 | M5 | à venir |
| Harnais, métriques, rapports | M6 | à venir |
| DIET, FunctionGemma | M7 | à venir |
| Pipeline audio | M8 | à venir |
