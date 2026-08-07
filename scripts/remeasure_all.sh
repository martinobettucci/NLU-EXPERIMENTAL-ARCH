#!/usr/bin/env bash
# Rejeu des campagnes restantes apres la correction de la passe de chauffe.
#
# Sequentiel par construction : les latences publiees seraient inexploitables
# si deux campagnes partageaient les memes coeurs. Le verrou exclusif du
# harnais l'imposerait de toute facon, mais l'ordre choisi ici va du moins
# couteux au plus couteux, pour que les resultats arrivent progressivement.
#
# Le depot doit etre propre au lancement : un fichier non suivi suffit a
# marquer chaque campagne `dirty`, donc impubliable. C'est deja arrive une
# fois, pour un telechargement lance pendant la sequence.
set -u

BENCH=".venv/bin/ivr-bench"
LOG="results/remeasure.log"

if [ -n "$(git status --porcelain -- ':!results/runs')" ]; then
  echo "depot modifie : les campagnes seraient marquees dirty. Committez d'abord." >&2
  git status --short -- ':!results/runs' >&2
  exit 2
fi

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
