#!/usr/bin/env bash
# Rejeu complet des campagnes apres la correction de la passe de chauffe.
#
# Sequentiel par construction : les latences publiees seraient inexploitables
# si deux campagnes partageaient les memes coeurs. Le verrou exclusif du
# harnais l'imposerait de toute facon, mais l'ordre choisi ici va du moins
# couteux au plus couteux, pour que les resultats arrivent progressivement.
set -u

BENCH=".venv/bin/ivr-bench"
LOG="results/remeasure.log"

full=(
  rules
  lexical_classifier
  diet
  embedding_classifier
  nearest_neighbour
  classifier_enum
  classifier_diet
  classifier_diet_rules
  classifier_diet_arbitrated
  classifier_diet_enum
)

# Architectures a plusieurs secondes par enonce : echantillon stratifie de 12
# cas par fonction, comme les campagnes publiees precedemment.
sampled=(
  functiongemma_zero_shot
  hybrid_functiongemma_top2
  needle_full
  hybrid_needle_top2
  hybrid_adaptive
  hypothetical_delta_top2
)

: > "$LOG"
for name in "${full[@]}"; do
  echo "=== $name (corpus complet) $(date -u +%H:%M:%S) ===" >> "$LOG"
  "$BENCH" benchmark text --architectures "$name" --seed 42 >> "$LOG" 2>&1 \
    || echo "!!! echec $name" >> "$LOG"
done

for name in "${sampled[@]}"; do
  echo "=== $name (12 par fonction) $(date -u +%H:%M:%S) ===" >> "$LOG"
  "$BENCH" benchmark text --architectures "$name" --seed 42 --per-function 12 >> "$LOG" 2>&1 \
    || echo "!!! echec $name" >> "$LOG"
done

# Le plus long en dernier : environ mille secondes d'inference.
echo "=== embedding_only (corpus complet) $(date -u +%H:%M:%S) ===" >> "$LOG"
"$BENCH" benchmark text --architectures embedding_only --seed 42 >> "$LOG" 2>&1 \
  || echo "!!! echec embedding_only" >> "$LOG"

echo "=== termine $(date -u +%H:%M:%S) ===" >> "$LOG"
