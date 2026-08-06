# Références et versions consultées

Les éléments ci-dessous ont été vérifiés directement contre les registres publics à la date
indiquée. Les versions comptent : une comparaison d'architectures n'a de sens que si l'on sait
exactement ce qui a été exécuté.

Date de consultation : **6 août 2026**.

## Modèles

| Modèle | Identifiant | Constat à la date de consultation |
|---|---|---|
| Needle | `Cactus-Compute/needle` | 26M paramètres, encodeur-décodeur, licence MIT, poids ouverts. Implémentation de référence en JAX : `github.com/cactus-compute/needle`, commit `6fdddb874dd6f46dabeea88d0bf239fc4a2474c4`. Architecture « Simple Attention Network », arXiv 2607.18363 |
| FunctionGemma | `google/functiongemma-270m-it` | Sous licence Gemma, accès conditionné à un jeton. **Le dépôt `google/functiongemma-270m` sans suffixe n'existe pas** (404). Format de contrôle : `<start_function_call>call:nom{arg:<escape>valeur<escape>}<end_function_call>`. Présenté par l'éditeur comme une base destinée à être spécialisée, y compris pour le multi-tours |
| EmbeddingGemma | `google/embeddinggemma-300m` | Sous licence Gemma. Dimension native 768, réductions Matryoshka documentées à 512, 256 et 128. **64 dimensions ne fait pas partie des sorties prévues** : elle exige donc une projection apprise sur l'entraînement, jamais une troncature arbitraire |
| Supertonic 3 | `Supertone/supertonic-3` | Synthèse vocale ONNX locale, licence OpenRAIL, 31 langues dont le français, 10 styles de voix préréglés (F1–F5, M1–M5) |
| Reconnaissance vocale | `Systran/faster-whisper-small` | `faster-whisper` 1.2.1 |

## Bibliothèques

| Bibliothèque | Version | Remarque |
|---|---|---|
| Rasa | 3.6.21 | `requires_python <3.11`. Incompatible avec l'environnement principal en Python 3.12, d'où le sidecar dédié. Le pipeline NLU de Rasa définit l'intention et les entités comme sorties centrales : c'est cette structure que l'adaptateur convertit vers le schéma de fonction canonique |
| transformers | 5.14.1 | `NeedleForCausalLM` n'est pas une architecture du tronc commun : Needle passe par son implémentation JAX de référence |
| `cactus-compute` | 2.0.1 | Runtime natif. Roues publiées pour macOS arm64 et manylinux aarch64 uniquement — **pas de roue x86_64**. Les latences Needle publiées ici sont donc mesurées sous JAX et constituent un plancher pessimiste |

## Méthodologie

- **Berkeley Function Calling Leaderboard** — source d'inspiration méthodologique pour la
  distinction entre appel simple, appels parallèles et enchaînements multiétapes. Il ne
  remplace pas le banc d'essai métier : les fonctions mesurées ici sont celles d'un serveur
  vocal de prise de rendez-vous, pas un catalogue générique.
- **CNIL, données de santé** — les données de rendez-vous et les motifs de consultation
  relèvent de données personnelles sensibles. Le dépôt reste synthétique et documente la
  minimisation des données, la durée de conservation, l'information des personnes et les
  responsabilités de traitement avant toute adaptation réelle. Voir `safety.md`.

## Matériel de référence des premières campagnes

x86_64, 4 cœurs, 15 Go de mémoire, **aucun accélérateur**. Les mesures de latence et de
mémoire ne valent que pour cette configuration et sont republiées avec chaque run.
