# Spécification du dépôt GitHub

## Nom de travail

`voice-intent-routing-benchmark`

Le nom doit rester configurable avant création du dépôt. Le README doit présenter le projet comme un banc d’essai reproductible pour le routage d’intentions et l’appel de fonctions dans un serveur vocal interactif francophone.

## 1. Mission

Créer un dépôt GitHub expérimental, exécutable et reproductible qui compare plusieurs architectures capables de transformer une demande vocale en une fonction métier structurée.

Le cas d’usage est un serveur vocal interactif destiné à un cabinet, un centre ou une plateforme de prise de rendez-vous médicaux. Le système doit pouvoir :

1. fournir des informations générales non cliniques ;
2. consulter les rendez-vous du patient authentifié ;
3. demander un nouveau rendez-vous avec un praticien nommé oralement par l’appelant ;
4. demander la modification d’un rendez-vous existant ;
5. reconnaître une demande hors périmètre ;
6. déclencher un transfert sécurisé lorsque la demande est ambiguë, sensible ou potentiellement urgente.

Le dépôt n’a pas pour objectif de produire un assistant médical, un outil de diagnostic ou un système de recommandation clinique. Toute question clinique doit être refusée ou transférée selon une politique déterministe.

## 2. Résultat attendu

L’agent de codage doit :

1. créer et structurer le dépôt ;
2. implémenter toutes les architectures définies dans cette spécification ;
3. générer et versionner les jeux de données synthétiques ;
4. entraîner les composants qui nécessitent un entraînement ;
5. construire les index vectoriels ;
6. exécuter les benchmarks texte, audio et bout en bout ;
7. produire des résultats bruts, des tableaux, des graphiques et des intervalles de confiance ;
8. mettre automatiquement à jour une section dédiée du `README.md` ;
9. documenter exactement le matériel, les versions, les paramètres et le commit utilisés ;
10. exécuter les tests et ne publier aucun résultat provenant d’une exécution incomplète ou non reproductible.

Le README ne doit contenir aucun résultat inventé. Tant qu’une suite n’a pas été exécutée, la cellule correspondante doit afficher `non exécuté`.

## 3. Hypothèse expérimentale principale

L’hypothèse à tester est la suivante :

> Une architecture hybride qui précompile des formulations hypothétiques par fonction, les encode dans un espace vectoriel compact, récupère deux fonctions candidates au runtime, puis confie uniquement ces fonctions à un micro-modèle d’appel d’outils peut atteindre une précision comparable ou supérieure à un appel direct sur le catalogue complet, avec une latence, une mémoire et un coût d’inférence inférieurs.

Le benchmark doit également vérifier les hypothèses secondaires suivantes :

1. le préfiltrage sémantique améliore la précision de Needle lorsque plusieurs fonctions ont des descriptions proches ;
2. le rappel top 2 du retriever est plus important que sa précision top 1, puisque la décision finale est confiée au modèle d’appel de fonctions ;
3. les embeddings de 128 dimensions conservent suffisamment d’information pour ce domaine ;
4. 64 dimensions peuvent être suffisantes pour un petit catalogue, mais doivent être évaluées et non supposées ;
5. la moyenne de tous les embeddings d’une fonction est moins robuste que plusieurs prototypes représentant ses différents modes sémantiques ;
6. les exemples contrastifs améliorent les frontières entre `prendre_rendez_vous`, `modifier_rendez_vous` et `consulter_rendez_vous` ;
7. la résolution du nom du praticien doit rester indépendante de la sélection de fonction ;
8. la dégradation due à la reconnaissance vocale doit être mesurée séparément de la qualité du routeur.

## 4. Principe de séparation des responsabilités

Le système doit distinguer quatre sous-problèmes :

1. reconnaissance vocale ;
2. sélection de la fonction métier ;
3. extraction des arguments exprimés dans la demande ;
4. orchestration du dialogue et exécution métier.

Le benchmark principal ne doit pas demander aux modèles de réaliser eux-mêmes une chaîne d’appels dépendants.

Exemple :

```text
« Je voudrais un rendez-vous avec le docteur Bensaïd mardi matin. »
```

La sortie attendue du routeur est une fonction métier atomique :

```json
{
  "name": "request_new_appointment",
  "arguments": {
    "practitioner_name": "docteur Bensaïd",
    "preferred_date": "mardi",
    "preferred_time": "matin"
  }
}
```

Le moteur déterministe réalise ensuite :

```text
résolution du praticien
→ recherche des disponibilités
→ proposition d’un créneau
→ confirmation explicite
→ création effective
```

Cette séparation est obligatoire. Elle permet une comparaison équitable entre DIET, Needle, FunctionGemma et l’architecture hybride. Les capacités de planification multiétape doivent être évaluées dans une suite secondaire distincte.

## 5. Catalogue métier canonique

Le catalogue doit être défini une seule fois dans `config/domain/functions.yaml`. Tous les adaptateurs doivent être générés à partir de cette source canonique.

### 5.1 `answer_general_information`

Fournit uniquement des informations administratives et pratiques.

Arguments :

```yaml
topic:
  type: string
  enum:
    - opening_hours
    - address
    - access
    - parking
    - accessibility
    - contact
    - pricing
    - payment
    - accepted_insurance
    - required_documents
    - appointment_preparation
    - teleconsultation
    - delays
    - other
```

### 5.2 `list_appointments`

Consulte les rendez-vous du patient déjà authentifié dans la session.

Arguments :

```yaml
date_from:
  type: string
  nullable: true
date_to:
  type: string
  nullable: true
practitioner_name:
  type: string
  nullable: true
```

Le `patient_id` ne doit jamais être extrait de la parole. Il est injecté par le contexte de session après authentification.

### 5.3 `request_new_appointment`

Démarre un workflow de prise de rendez-vous.

Arguments :

```yaml
practitioner_name:
  type: string
  nullable: true
specialty:
  type: string
  nullable: true
preferred_date:
  type: string
  nullable: true
preferred_time:
  type: string
  nullable: true
reason:
  type: string
  nullable: true
```

Le champ `reason` reste facultatif. Il ne doit pas être persisté dans les résultats de benchmark.

### 5.4 `request_appointment_reschedule`

Démarre un workflow de modification d’un rendez-vous existant.

Arguments :

```yaml
appointment_reference:
  type: string
  nullable: true
practitioner_name:
  type: string
  nullable: true
current_date:
  type: string
  nullable: true
preferred_new_date:
  type: string
  nullable: true
preferred_new_time:
  type: string
  nullable: true
```

### 5.5 `emergency_handoff`

Interrompt le workflow normal et transfère vers une consigne ou un opérateur prévu par la politique de sécurité.

Arguments :

```yaml
reason_category:
  type: string
  enum:
    - explicit_emergency
    - severe_symptoms
    - immediate_danger
    - unclear_health_risk
```

Le modèle ne doit pas diagnostiquer. Il doit seulement reconnaître qu’un traitement administratif ordinaire n’est pas approprié.

### 5.6 `human_handoff`

Transfère la demande vers un humain.

Arguments :

```yaml
reason:
  type: string
  enum:
    - ambiguous_request
    - unsupported_request
    - repeated_failure
    - practitioner_not_resolved
    - authentication_required
    - user_requested_human
```

### 5.7 `no_tool`

Pseudo-fonction utilisée uniquement par le benchmark pour mesurer le rejet correct des demandes sans outil applicable.

Cette fonction ne doit jamais être exécutée dans le backend métier.

## 6. Backend métier simulé

Le dépôt doit fournir un backend entièrement synthétique, reproductible et auto-initialisé.

Technologie par défaut :

```text
Python 3.12
FastAPI
Pydantic
SQLite pour le profil local
PostgreSQL pour le profil benchmark optionnel
```

Le backend doit contenir :

1. des patients synthétiques ;
2. des praticiens synthétiques ;
3. des spécialités ;
4. des sites ;
5. des créneaux disponibles ;
6. des rendez-vous existants ;
7. des informations générales ;
8. des identifiants opaques ;
9. une horloge figée et configurable pour rendre les dates relatives reproductibles.

Aucune donnée réelle ne doit être utilisée.

Les opérations de création et de modification doivent fonctionner en mode transactionnel simulé. Chaque benchmark doit réinitialiser l’état ou utiliser un mode `dry_run` afin qu’un test ne modifie pas les résultats des tests suivants.

## 7. Catalogue synthétique des praticiens

Créer au minimum 1 000 praticiens synthétiques.

Le catalogue doit contenir :

```yaml
practitioner_id:
display_name:
first_name:
last_name:
specialty:
site_id:
aliases:
phonetic_aliases:
```

Le générateur doit couvrir :

1. noms français fréquents ;
2. noms internationaux ;
3. noms avec accents ;
4. noms avec apostrophes ;
5. noms composés ;
6. noms courts ;
7. homophones ;
8. paires proches, par exemple `Rey`, `Ray`, `Reï` ;
9. noms susceptibles d’être mal transcrits par un ASR ;
10. titres exprimés de différentes manières, par exemple `docteur`, `docteure`, `Dr`, `madame le docteur`.

La partition de test doit inclure des praticiens absents des exemples d’entraînement. Le routeur doit extraire le texte prononcé, pas mémoriser une liste fermée de personnes.

## 8. Résolution du praticien

La résolution du praticien est un composant séparé du routeur.

Entrée :

```json
{
  "practitioner_name": "docteur ben said"
}
```

Sortie :

```json
{
  "status": "resolved",
  "practitioner_id": "practitioner_00421",
  "confidence": 0.94,
  "candidates": [
    {
      "practitioner_id": "practitioner_00421",
      "display_name": "Dr Nadia Bensaïd",
      "score": 0.94
    }
  ]
}
```

Statuts autorisés :

```text
resolved
ambiguous
not_found
missing
```

Le composant doit combiner au minimum :

1. normalisation Unicode ;
2. suppression contrôlée des titres ;
3. comparaison lexicale ;
4. comparaison phonétique ;
5. alias ;
6. spécialité ou site lorsqu’ils sont présents dans la demande ;
7. seuil de clarification.

Le modèle d’appel de fonctions ne doit jamais générer directement un `practitioner_id`.

## 9. Architectures à comparer

Toutes les architectures doivent implémenter la même interface Python :

```python
class Router(Protocol):
    def predict(
        self,
        utterance: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> RouterPrediction:
        ...
```

La sortie canonique est :

```python
class RouterPrediction(BaseModel):
    tool_name: str | None
    arguments: dict[str, object]
    confidence: float | None
    candidates: list[ToolCandidate]
    raw_output: str | dict | None
    latency_ms: float
    metadata: dict[str, object]
```

### A0. Baseline déterministe

Règles, expressions régulières et listes de mots clés.

Objectif : établir un plancher de performance et vérifier que les jeux de données ne sont pas trivialement séparables.

### A1. DIET

Rasa DIET entraîné sur les mêmes formulations synthétiques utilisées par les autres architectures.

Sorties :

```text
intent
entities
confidence
```

Un adaptateur déterministe convertit l’intention et les entités vers le schéma de fonction canonique.

Configurations minimales :

1. DIET léger avec caractéristiques lexicales ;
2. DIET avec extraction d’entités ;
3. optionnellement DIET avec un featurizer linguistique externe, publié séparément car l’empreinte n’est plus comparable.

### A2. Needle, catalogue complet

Needle reçoit la demande initiale et toutes les définitions de fonctions.

Aucun entraînement métier obligatoire.

Le prompt, le format des outils, la quantification et le runtime doivent être versionnés.

### A3. FunctionGemma, catalogue complet, zéro-shot

FunctionGemma reçoit la demande initiale et toutes les définitions.

Le format officiel de déclaration et d’appel de fonctions doit être respecté.

### A4. FunctionGemma spécialisé

Fine-tuning supervisé sur le même corpus métier.

Cette architecture doit être publiée séparément de la version zéro-shot. Le temps d’entraînement, le matériel, l’énergie si mesurable et la taille de l’adaptateur ou du checkpoint doivent être rapportés.

### A5. Retriever sémantique seul

Recherche sémantique sur les formulations hypothétiques.

La fonction agrégée la mieux classée est directement sélectionnée. Les arguments sont extraits par des règles ou laissés non évalués dans une sous-suite dédiée.

Cette baseline mesure la valeur du retriever avant l’ajout de Needle.

### A6. Retriever sémantique top 2 puis Needle

Architecture principale proposée.

Runtime :

```text
demande
→ embedding compact
→ recherche des prototypes
→ agrégation par fonction
→ deux fonctions candidates
→ Needle avec seulement ces deux définitions
→ validation du schéma
```

La phrase originale, et non une phrase synthétique voisine, doit être transmise à Needle.

### A7. Retriever sémantique top 2 puis FunctionGemma

Même pipeline que A6 avec FunctionGemma comme arbitre final.

### A8. Variante adaptative

Le nombre de fonctions transmis au modèle dépend de l’incertitude :

```text
confiance forte et marge forte → 1 fonction
ambiguïté locale → 2 fonctions
ambiguïté élevée → 3 ou 4 fonctions
score faible → no_tool ou human_handoff
```

Cette variante doit être comparée séparément au top 2 fixe.

## 10. Construction offline de l’index sémantique

### 10.1 Source canonique

Pour chaque fonction, le fichier `functions.yaml` contient :

```yaml
name:
description:
parameters:
positive_seeds:
confusable_with:
unsupported_examples:
safety_notes:
```

### 10.2 Génération hypothétique

Un générateur configurable produit des formulations qui devraient appeler chaque fonction.

Chaque fonction doit recevoir au minimum :

```text
300 formulations positives pour l’index
100 formulations contrastives
100 formulations de validation
300 formulations de test provenant d’un autre protocole de génération
```

Les formulations doivent couvrir :

1. phrases complètes ;
2. commandes courtes ;
3. hésitations ;
4. répétitions ;
5. langage familier ;
6. fautes grammaticales ;
7. dates relatives ;
8. dates absolues ;
9. heures ;
10. absence d’arguments ;
11. arguments multiples ;
12. négations ;
13. corrections ;
14. noms de praticiens ;
15. spécialités ;
16. demandes ambiguës ;
17. demandes hors périmètre ;
18. demandes potentiellement urgentes.

### 10.3 Prévention des fuites

Le dépôt doit empêcher qu’une paraphrase presque identique apparaisse dans l’index et dans le test.

Mesures obligatoires :

1. séparation par gabarit sémantique, pas seulement par ligne ;
2. graines différentes ;
3. prompt de génération différent pour le test ;
4. modèle générateur différent lorsque l’infrastructure le permet ;
5. déduplication lexicale ;
6. déduplication par similarité sémantique ;
7. rapport de contamination ;
8. jeu de test manuel minimal versionné.

Le rapport doit afficher le nombre d’éléments supprimés comme doublons.

### 10.4 Embeddings

Le système doit fournir une interface d’encodeur interchangeable.

Configurations obligatoires :

```text
64 dimensions
128 dimensions
256 dimensions
dimension native du modèle
```

Une dimension réduite ne doit être utilisée que si :

1. le modèle a été entraîné avec une représentation de type Matryoshka ;
2. une projection a été apprise sur le jeu d’entraînement ;
3. une réduction comme PCA a été ajustée uniquement sur les données d’entraînement.

La troncature arbitraire d’un embedding non prévu pour cela est interdite.

Au moins un modèle multilingue compatible avec 128 dimensions doit être benchmarké. EmbeddingGemma peut servir de référence MRL à 128 dimensions, mais le dépôt doit aussi permettre un encodeur plus petit afin de vérifier que le coût du retriever ne dépasse pas celui du routeur final.

### 10.5 Normalisation

Tous les vecteurs destinés à la similarité cosinus doivent être normalisés L2 une seule fois.

Le runtime ne doit pas recalculer les embeddings offline.

### 10.6 Prototypes

Comparer au minimum :

1. tous les vecteurs ;
2. un seul centroïde par fonction ;
3. 4 prototypes par fonction ;
4. 8 prototypes par fonction ;
5. 16 prototypes par fonction ;
6. sélection par clustering ;
7. sélection par medoids.

Le benchmark doit vérifier l’hypothèse selon laquelle un seul centroïde détruit les modes sémantiques minoritaires.

## 11. Recherche et agrégation

Pour un petit index, le produit matriciel exact doit être la baseline. FAISS ou un autre index ANN ne doit être activé qu’en variante.

Pipeline recommandé :

```text
embedding de la demande
→ top 30 prototypes
→ regroupement par function_id
→ calcul d’un score par fonction
→ classement des fonctions
```

Score par défaut, configurable et calibré uniquement sur le jeu de validation :

```text
score fonction =
0,55 × meilleure similarité
+ 0,35 × moyenne des trois meilleurs voisins
+ 0,10 × soutien du voisinage
```

Le soutien du voisinage doit être normalisé entre 0 et 1.

Mesures d’incertitude :

```text
score absolu du premier candidat
delta entre le premier et le deuxième
pureté des voisins
entropie de la distribution des fonctions
```

Le delta cosinus doit être traité comme un signal d’incertitude, pas comme l’unique score de classement.

## 12. Arguments et valeurs manquantes

Une demande peut sélectionner correctement une fonction sans fournir tous les arguments.

Exemple :

```text
« Je voudrais prendre rendez-vous. »
```

Sortie correcte :

```json
{
  "name": "request_new_appointment",
  "arguments": {
    "practitioner_name": null,
    "specialty": null,
    "preferred_date": null,
    "preferred_time": null,
    "reason": null
  }
}
```

Le routeur ne doit pas inventer les valeurs manquantes.

Le gestionnaire de dialogue doit produire la prochaine question déterministe :

```text
« Avec quel médecin ou pour quelle spécialité souhaitez-vous prendre rendez-vous ? »
```

Les métriques doivent pénaliser les arguments hallucinés plus fortement que les arguments absents.

## 13. Gestion du contexte conversationnel

La suite principale est single-turn.

Une suite secondaire doit tester le remplissage de slots sur plusieurs tours avec un gestionnaire d’état déterministe.

Exemple :

```text
Utilisateur : Je veux prendre rendez-vous avec le docteur Rey.
Système : Quel jour vous conviendrait ?
Utilisateur : Mardi matin.
```

Le second message doit être interprété dans le contexte actif, sans demander au routeur de reclasser arbitrairement la demande comme une nouvelle intention.

Le contexte doit contenir :

```yaml
active_workflow:
pending_slots:
confirmed_slots:
candidate_practitioners:
patient_id:
locale:
timezone:
```

Fuseau horaire par défaut :

```text
Europe/Paris
```

## 14. Pipeline vocal

Le pipeline vocal doit être modulaire :

```text
audio
→ VAD
→ ASR
→ normalisation minimale
→ routeur
→ gestionnaire de dialogue
→ backend simulé
→ génération de réponse
→ TTS
```

L’ASR et le TTS ne doivent pas être imposés au niveau du cœur du benchmark. Ils sont implémentés derrière des adaptateurs.

Le benchmark doit posséder trois modes :

### 14.1 Mode texte oracle

Entrée : transcription de référence.

Objectif : mesurer le routeur sans erreur vocale.

### 14.2 Mode transcription ASR

Entrée : transcription produite à partir de l’audio.

Objectif : mesurer la dégradation introduite par l’ASR.

### 14.3 Mode bout en bout

Entrée : audio.

Sortie : action métier et réponse synthétisée.

Objectif : mesurer le taux de réussite réel et la latence totale.

## 15. Jeux audio

Créer un jeu audio reproductible à partir des textes de test.

Le générateur doit produire plusieurs variantes :

1. voix masculines ;
2. voix féminines ;
3. vitesses différentes ;
4. pauses ;
5. hésitations ;
6. bruit de bureau ;
7. bruit de rue modéré ;
8. compression téléphonique ;
9. bande passante 8 kHz ;
10. volume faible ;
11. accents francophones disponibles ;
12. noms propres épelés ;
13. correction d’un nom après une mauvaise reconnaissance.

Les fichiers doivent utiliser des identifiants synthétiques et être accompagnés d’un manifeste.

Un sous-ensemble humain réel pourra être ajouté ultérieurement, avec consentement et gouvernance distincte. Il ne doit pas être requis pour reproduire la première version.

## 16. Cas de test obligatoires

### 16.1 Informations générales

```text
À quelle heure ouvre le cabinet ?
Où est le parking ?
Est-ce accessible en fauteuil roulant ?
Quels papiers dois-je apporter ?
Est-ce que je peux payer par carte ?
```

### 16.2 Consultation

```text
Quels sont mes prochains rendez-vous ?
J’ai quoi de prévu la semaine prochaine ?
Est-ce que j’ai déjà un rendez-vous avec le docteur Martin ?
```

### 16.3 Nouveau rendez-vous

```text
Je voudrais voir le docteur Bensaïd mardi matin.
Prenez-moi un rendez-vous avec le docteur Rey.
J’ai besoin d’un dermatologue jeudi après-midi.
Je veux un rendez-vous, peu importe le médecin.
```

### 16.4 Modification

```text
Déplacez mon rendez-vous de demain à vendredi.
Je veux modifier celui avec le docteur Martin.
Gardez le même médecin mais mettez-le l’après-midi.
```

### 16.5 Négations et contrastes

```text
Je ne veux pas annuler, je veux juste changer l’heure.
Je ne veux pas modifier mon rendez-vous, je veux seulement savoir quand il est.
Je ne cherche pas un généraliste, je veux le docteur Rey précisément.
```

### 16.6 Hors périmètre

```text
Quel temps fera-t-il demain ?
Pouvez-vous appeler ma banque ?
Racontez-moi une histoire.
Annulez mon abonnement téléphonique.
```

### 16.7 Sécurité

```text
J’ai très mal à la poitrine et je respire mal.
C’est urgent, je viens de perdre connaissance.
Je veux un diagnostic.
Quel médicament dois-je prendre ?
```

Le système ne doit produire aucune recommandation médicale.

### 16.8 ASR et praticiens

```text
docteur Rey / Ray / Reï
docteur Bensaïd / Ben Saïd / Ben Said
docteur D’Angelo
docteur Jean-Pierre Le Goff
docteur Nguyen
```

## 17. Métriques de qualité

### 17.1 Routage

Métriques obligatoires :

```text
tool accuracy
macro F1 par fonction
balanced accuracy
top 2 recall du retriever
top 3 recall du retriever
no_tool precision
no_tool recall
emergency_handoff recall
human_handoff precision
matrice de confusion
```

La métrique de sécurité prioritaire est le rappel de `emergency_handoff`.

### 17.2 Arguments

```text
JSON parse success
JSON Schema validity
exact tool call match
argument exact match
argument micro F1
argument macro F1
required argument recall
hallucinated argument rate
missing argument rate
date normalization accuracy
time normalization accuracy
practitioner span accuracy
```

### 17.3 Praticien

```text
resolver top 1 accuracy
resolver top 3 recall
ambiguity detection precision
ambiguity detection recall
clarification rate
false unique resolution rate
```

Une résolution erronée avec forte confiance doit être considérée comme plus grave qu’une clarification.

### 17.4 Voix

```text
WER global
CER global
WER sur noms de praticiens
CER sur noms de praticiens
routing degradation from oracle transcript
argument degradation from oracle transcript
end-to-end task success
```

### 17.5 Performance

Mesurer séparément chaque étape :

```text
cold start
ASR
embedding
vector search
candidate aggregation
tool caller prefill
tool caller decode
schema validation
practitioner resolution
business workflow
TTS
end-to-end
```

Rapporter :

```text
p50
p95
p99
mean
standard deviation
throughput
peak RSS
steady RSS
model disk size
index disk size
CPU utilization
GPU utilization when applicable
VRAM when applicable
```

Les résultats CPU et GPU doivent apparaître dans des tableaux séparés.

### 17.6 Coût d’adaptation

Pour chaque architecture :

```text
nombre d’exemples
temps de génération
temps d’entraînement
temps de construction de l’index
taille du corpus
taille des poids
taille de l’index
temps d’ajout d’une nouvelle fonction
réentraînement global requis ou non
```

## 18. Protocole statistique

Exigences :

1. au moins 5 graines pour les composants entraînés ;
2. intervalles de confiance bootstrap à 95 % ;
3. ordre des exemples randomisé avec graine enregistrée ;
4. comparaison appariée entre architectures ;
5. test de McNemar pour les différences de réussite binaire lorsque pertinent ;
6. publication du nombre exact d’exemples ;
7. publication des exclusions ;
8. publication des erreurs ;
9. aucun agrégat sans résultat par catégorie.

Le classement principal doit suivre cet ordre :

1. taux de réussite métier bout en bout ;
2. exactitude de la fonction ;
3. rappel de sécurité ;
4. validité des arguments ;
5. latence p95 ;
6. mémoire maximale.

Ne pas masquer les compromis dans un score composite unique. Une vue de Pareto doit être produite.

## 19. Robustesse

Créer des sous-suites dédiées :

```text
clean_text
typos
spoken_disfluencies
negation
hard_contrast
missing_arguments
multi_intent
no_tool
safety
doctor_names
asr_clean
asr_noisy
telephone_8khz
```

Chaque résultat global doit pouvoir être décomposé par sous-suite.

## 20. Interface CLI

Fournir une CLI avec Typer ou équivalent.

Commandes minimales :

```bash
ivr-bench data generate
ivr-bench data validate
ivr-bench doctors generate
ivr-bench diet train
ivr-bench index build
ivr-bench benchmark text
ivr-bench benchmark audio
ivr-bench benchmark e2e
ivr-bench benchmark all
ivr-bench report build
ivr-bench readme update
ivr-bench reproduce
```

Exemple :

```bash
uv run ivr-bench benchmark text \
  --architectures diet,needle_full,hybrid_needle_top2 \
  --config config/benchmark/cpu.yaml \
  --seed 42
```

## 21. Commandes Make

```bash
make setup
make lint
make typecheck
make test
make generate-data
make train
make build-index
make benchmark-smoke
make benchmark-text
make benchmark-audio
make benchmark-full
make report
make update-readme
make reproduce
```

Ajouter :

```bash
./runDev
./runBenchmark
./runProd
```

`runProd` démarre uniquement la démonstration du serveur vocal simulé. Le dépôt reste d’abord un projet expérimental.

## 22. Structure du dépôt

```text
.
├── README.md
├── SPECIFICATION.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── uv.lock
├── Makefile
├── runDev
├── runBenchmark
├── runProd
├── docker-compose.dev.yml
├── docker-compose.benchmark.yml
├── docker-compose.prod.yml
├── config
│   ├── domain
│   │   ├── functions.yaml
│   │   ├── general_information.yaml
│   │   └── safety_policy.yaml
│   ├── architectures
│   │   ├── rules.yaml
│   │   ├── diet.yaml
│   │   ├── needle_full.yaml
│   │   ├── functiongemma_zero_shot.yaml
│   │   ├── functiongemma_tuned.yaml
│   │   ├── embedding_only.yaml
│   │   ├── hybrid_needle_top2.yaml
│   │   ├── hybrid_functiongemma_top2.yaml
│   │   └── hybrid_adaptive.yaml
│   ├── embeddings
│   ├── asr
│   ├── tts
│   └── benchmark
├── data
│   ├── seeds
│   ├── generated
│   │   ├── train
│   │   ├── validation
│   │   ├── index
│   │   └── test
│   ├── audio
│   ├── doctors
│   ├── manifests
│   └── schemas
├── src
│   └── ivr_bench
│       ├── cli
│       ├── domain
│       ├── generators
│       ├── routers
│       │   ├── rules
│       │   ├── diet
│       │   ├── needle
│       │   ├── functiongemma
│       │   └── hybrid
│       ├── embeddings
│       ├── retrieval
│       ├── resolver
│       ├── dialogue
│       ├── asr
│       ├── tts
│       ├── backend
│       ├── benchmark
│       ├── metrics
│       ├── reporting
│       └── api
├── tests
│   ├── unit
│   ├── integration
│   ├── contract
│   ├── regression
│   └── smoke
├── results
│   ├── runs
│   ├── latest
│   ├── tables
│   ├── charts
│   └── errors
├── reports
├── scripts
├── docs
└── .github
    └── workflows
        ├── ci.yml
        ├── benchmark-smoke.yml
        ├── benchmark-full.yml
        └── publish-results.yml
```

## 23. Format des données

Chaque cas doit utiliser un schéma JSONL versionné.

Exemple :

```json
{
  "id": "test_create_000123",
  "split": "test",
  "suite": "hard_contrast",
  "language": "fr",
  "utterance": "Je ne veux pas voir n’importe qui, prenez-moi le docteur Rey mardi.",
  "expected": {
    "tool_name": "request_new_appointment",
    "arguments": {
      "practitioner_name": "docteur Rey",
      "preferred_date": "mardi"
    }
  },
  "metadata": {
    "generator": "generator_b",
    "template_family": "explicit_practitioner_negated_alternative",
    "contains_sensitive_synthetic_data": false
  }
}
```

Un manifeste doit contenir :

```text
hash du fichier
nombre de lignes
répartition par fonction
répartition par sous-suite
version du schéma
version du générateur
date de génération
```

## 24. Validation des sorties

Toutes les sorties de modèles doivent passer par :

```text
parseur strict
→ normalisation minimale
→ validation Pydantic
→ validation JSON Schema
→ validation métier
```

Le parseur ne doit pas corriger silencieusement une sortie invalide au point de masquer une erreur du modèle.

Rapporter séparément :

```text
sortie valide nativement
sortie récupérable avec réparation déterministe mineure
sortie invalide
```

Les résultats principaux utilisent la sortie valide nativement. Les résultats après réparation apparaissent dans une colonne distincte.

## 25. Politique de sécurité

Les règles de sécurité doivent être implémentées hors modèle.

Exigences :

1. aucune prise ou modification effective sans confirmation explicite ;
2. aucune consultation sans contexte patient authentifié ;
3. aucun `patient_id` extrait de la voix ;
4. aucun identifiant de praticien inventé ;
5. aucun conseil médical ;
6. transfert déterministe pour urgence ou risque potentiel ;
7. journalisation sans contenu vocal brut par défaut ;
8. pseudonymisation des identifiants ;
9. masquage des noms et motifs dans les traces publiées ;
10. rétention configurable ;
11. secret absent du dépôt ;
12. tests de non-régression sur les scénarios de sécurité.

Les données de rendez-vous et les éventuels motifs peuvent relever des données de santé. Le dépôt doit rester synthétique et documenter la minimisation des données, la durée de conservation, l’information des personnes et les responsabilités de traitement avant toute adaptation réelle.

## 26. Mesure de l’ajout d’une nouvelle fonction

Ajouter après la première campagne une fonction non présente initialement :

```text
request_appointment_cancellation
```

Mesurer pour chaque architecture :

```text
code modifié
données ajoutées
temps humain estimé
temps machine
réentraînement requis
réindexation requise
régression sur les fonctions existantes
précision sur la nouvelle fonction
```

Cette expérience mesure directement la flexibilité opérationnelle.

Pour l’architecture hybride :

```text
nouvelle définition
→ génération offline
→ embeddings
→ insertion dans l’index
```

Pour DIET :

```text
nouvelle intention
→ exemples
→ réentraînement
→ nouvelle évaluation complète
```

## 27. Expérience d’ablation de l’architecture hybride

Variantes obligatoires :

1. 64 contre 128 contre 256 dimensions ;
2. 1 contre 4 contre 8 contre 16 prototypes ;
3. centroïde contre medoids ;
4. formulations positives seules contre positives et contrastives ;
5. top 1 contre top 2 contre top 3 ;
6. top 2 fixe contre sélection adaptative ;
7. score maximum seul contre score agrégé ;
8. index exact contre ANN ;
9. phrase originale seule contre ajout des voisins synthétiques dans le prompt ;
10. Needle avec catalogue complet contre Needle avec candidats.

L’option ajoutant les voisins synthétiques au prompt ne doit pas être la valeur par défaut. Elle sert uniquement à vérifier si elle aide ou parasite la décision.

## 28. Reproductibilité

Chaque run doit enregistrer :

```yaml
run_id:
timestamp:
git_commit:
git_dirty:
os:
kernel:
python:
dependencies:
container_digest:
cpu:
cpu_threads:
ram:
gpu:
vram:
drivers:
runtime:
models:
quantization:
seeds:
dataset_hashes:
config_hash:
command:
duration:
```

Un run réalisé avec un dépôt modifié mais non commité doit être marqué `dirty` et ne doit pas remplacer les résultats officiels du README.

## 29. GitHub Actions

### 29.1 CI standard

À chaque pull request :

```text
format
lint
typecheck
unit tests
contract tests
dataset schema validation
smoke benchmark sur petit corpus
README generated-section consistency
```

### 29.2 Benchmark complet

Le workflow complet doit être déclenché manuellement ou sur runner auto-hébergé.

Paramètres de `workflow_dispatch` :

```text
suite
architectures
hardware_profile
seed_count
publish_results
```

Un runner GitHub hébergé ne doit pas être utilisé pour publier des comparaisons de performance matérielle comme si elles étaient stables.

### 29.3 Publication

Le workflow de publication doit :

1. vérifier que tous les runs requis sont complets ;
2. vérifier les hashes ;
3. générer les tableaux ;
4. générer les graphiques ;
5. mettre à jour le README entre deux marqueurs ;
6. créer un commit dédié ;
7. joindre les artefacts au workflow ;
8. ne jamais écraser les runs historiques.

## 30. Mise à jour automatique du README

Le README doit contenir :

```html
<!-- BENCHMARK_RESULTS_START -->
<!-- BENCHMARK_RESULTS_END -->
```

Le script `ivr-bench readme update` remplace uniquement cette section.

Contenu généré :

1. date et commit ;
2. matériel ;
3. versions des modèles ;
4. taille du dataset ;
5. tableau principal ;
6. résultats par sous-suite ;
7. latence ;
8. mémoire ;
9. empreinte disque ;
10. coût d’adaptation ;
11. ablations ;
12. erreurs principales ;
13. liens relatifs vers les résultats bruts ;
14. commande de reproduction.

Exemple de tableau :

```text
| Architecture | Tool accuracy | Argument F1 | E2E success | Safety recall | p95 CPU | Peak RSS | Disk |
```

Toute métrique absente doit afficher `non exécuté`, jamais `0`.

## 31. Graphiques obligatoires

Créer des graphiques statiques en PNG et SVG :

1. précision contre latence p95 ;
2. précision contre mémoire ;
3. succès bout en bout par architecture ;
4. rappel top k du retriever ;
5. impact de la dimension d’embedding ;
6. impact du nombre de prototypes ;
7. matrice de confusion ;
8. dégradation oracle contre ASR ;
9. précision de résolution des praticiens ;
10. coût d’ajout d’une fonction.

Les données sources doivent être enregistrées en CSV ou Parquet.

## 32. Tests

### 32.1 Unitaires

Tester :

```text
normalisation
agrégation
calcul du delta
sélection top k
validation des schémas
résolution des praticiens
dates relatives
gestion du contexte
mise à jour du README
```

### 32.2 Contractuels

Chaque routeur doit réussir les mêmes tests de contrat.

### 32.3 Intégration

Tester le chemin :

```text
texte
→ routeur
→ dialogue
→ backend
→ réponse
```

### 32.4 Régression

Conserver un petit corpus critique :

```text
négations
urgence
no_tool
noms proches
arguments absents
modification contre consultation
```

Une baisse sur le rappel de sécurité doit bloquer la publication.

## 33. Qualité du code

Exigences :

```text
Python typé strictement
Pydantic pour les contrats
Ruff
mypy ou pyright
pytest
coverage
logs structurés
configuration déclarative
aucune dépendance cachée
aucun chemin absolu
aucun téléchargement silencieux pendant les tests
```

Les modèles doivent être téléchargés par une commande explicite et mis en cache dans un volume configuré.

## 34. Profils Docker

### Dev

```text
auto-initialisé
jeu de données minimal
modèles mock ou très petits
tests rapides
aucun service externe obligatoire
```

### Benchmark

```text
données complètes
modèles réels
volumes de cache
mesures de ressources
CPU et GPU configurables
```

### Prod démonstration

```text
API
backend synthétique
pipeline vocal
aucune donnée réelle
aucune publication automatique
```

## 35. Documentation

Créer :

```text
README.md
SPECIFICATION.md
docs/architecture.md
docs/dataset.md
docs/benchmark_protocol.md
docs/safety.md
docs/reproducibility.md
docs/adding_an_architecture.md
docs/adding_a_function.md
docs/results_interpretation.md
```

Le README doit rester orienté exécution et résultats. Les détails méthodologiques vont dans `docs`.

## 36. Références techniques à documenter

Le dépôt doit citer dans sa documentation :

1. la documentation officielle de Rasa sur les intentions, les entités et DIET ;
2. le dépôt et la documentation officielle de Needle ;
3. la documentation officielle de FunctionGemma, notamment son format de contrôle et ses limites single-turn, parallel, multi-turn et multi-step ;
4. le Berkeley Function Calling Leaderboard comme inspiration méthodologique, sans remplacer le benchmark métier ;
5. la documentation officielle du modèle d’embedding retenu ;
6. les recommandations de la CNIL concernant les données de santé.

Les versions exactes consultées doivent être enregistrées dans `docs/references.md`.

## 37. Critères d’acceptation

Le travail est accepté lorsque :

1. `make setup` fonctionne sur une machine propre ;
2. `make test` réussit ;
3. `make benchmark-smoke` compare toutes les architectures avec des mocks ou petits checkpoints ;
4. `make benchmark-text` exécute la campagne texte complète ;
5. `make benchmark-audio` exécute la campagne audio ;
6. `make report` produit JSON, CSV, Parquet, PNG, SVG et Markdown ;
7. `make update-readme` met à jour uniquement la zone générée ;
8. les résultats sont traçables jusqu’au run brut ;
9. aucun résultat n’est saisi manuellement ;
10. le matériel et les versions sont présents ;
11. les intervalles de confiance sont calculés ;
12. les scénarios de sécurité sont visibles séparément ;
13. le dépôt contient une commande de reproduction ;
14. une nouvelle architecture peut être ajoutée via une interface documentée ;
15. une nouvelle fonction peut être ajoutée sans modifier le moteur hybride ;
16. l’expérience d’ajout de `request_appointment_cancellation` est exécutée ;
17. le README contient les résultats réellement obtenus ;
18. les limites et les échecs sont publiés, pas seulement les meilleurs scores.

## 38. Ordre d’exécution imposé à l’agent

1. créer le squelette du dépôt ;
2. implémenter les contrats et schémas ;
3. créer le backend synthétique ;
4. créer le générateur de praticiens ;
5. créer les données de référence ;
6. implémenter la baseline déterministe ;
7. implémenter DIET ;
8. implémenter Needle ;
9. implémenter FunctionGemma ;
10. implémenter le retriever et l’agrégation ;
11. implémenter les architectures hybrides ;
12. implémenter le résolveur de praticiens ;
13. implémenter le benchmark texte ;
14. implémenter le benchmark audio ;
15. implémenter le pipeline bout en bout ;
16. implémenter les métriques et statistiques ;
17. implémenter la génération des rapports ;
18. implémenter la mise à jour du README ;
19. exécuter les tests ;
20. exécuter les benchmarks ;
21. analyser automatiquement les erreurs ;
22. publier les résultats et commandes exactes dans le README.

## 39. Livrables finaux de l’agent

L’agent doit produire dans le dépôt :

```text
code source complet
tests
configurations
données synthétiques
modèles ou scripts de récupération
index vectoriels reproductibles
résultats bruts
résultats agrégés
graphiques
README mis à jour
documentation
journal des commandes exécutées
rapport des limites
```

Le dernier message de l’agent doit fournir :

```text
URL du dépôt
commit des résultats
commandes exécutées
tests réussis
benchmarks exécutés
matériel utilisé
principaux résultats
éléments non exécutés
limites connues
```
