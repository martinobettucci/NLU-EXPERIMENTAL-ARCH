# Sécurité et données

## Règles hors modèle

La politique de sécurité n'est jamais déléguée au modèle (§25) :

1. aucune prise ni modification effective sans confirmation explicite ;
2. aucune consultation sans contexte patient authentifié ;
3. aucun `patient_id` extrait de la parole — il vient de la session ;
4. aucun identifiant de praticien inventé par un modèle ;
5. aucun conseil médical, aucun diagnostic ;
6. transfert déterministe en cas d'urgence ou de risque potentiel.

Le modèle n'a pas à reconnaître une pathologie : il doit seulement reconnaître qu'un traitement
administratif ordinaire n'est pas approprié.

## Métrique de sécurité prioritaire

Le rappel de `emergency_handoff`. Une baisse sur cette métrique bloque la publication.

## Données de santé

Les rendez-vous et les motifs de consultation peuvent constituer des données de santé, donc des
données personnelles sensibles. Le dépôt reste entièrement synthétique. Le champ `reason` n'est
jamais persisté dans les résultats de benchmark. Les journaux ne contiennent pas d'audio brut
par défaut, les identifiants sont pseudonymisés et les noms sont masqués dans les traces
publiées. Toute adaptation à des données réelles exigerait au préalable une analyse de
minimisation, de durée de conservation, d'information des personnes et de responsabilité de
traitement — voir `references.md`.
