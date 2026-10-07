#!/usr/bin/env bash
# Runs the E2 workloads for each model; skips a run if another model server is
# on the GPU and stops if free disk space drops below 30 GB.
set -u
cd "$(dirname "$0")"
check() {
  if pgrep -f 'llama-server|bin/gufo serve' >/dev/null; then
    echo "SKIP $1: another model server is running"; return 1
  fi
  local free_gb
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$free_gb" -lt 30 ]; then echo "STOP: only ${free_gb}G free"; exit 1; fi
  echo "=== $1 (free ${free_gb}G) $(date +%T)"
}
for model in ${MODELS:-fn q27}; do
  check "$model w1" && { python3 w1_agent.py --model "$model" || echo "FAIL $model w1"; }
  for w in w2 w3 w4; do
    check "$model $w" && { python3 workloads.py "$w" --model "$model" || echo "FAIL $model $w"; }
  done
done
echo "=== done $(date +%T)"
