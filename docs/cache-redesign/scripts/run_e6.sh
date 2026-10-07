#!/usr/bin/env bash
# E6: today's system at concurrency 4 (4 sessions, 4 client workers).
# Falls back to a 65,536-token context if 131,072 does not start.
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
run() {  # model workload extra...
  local model=$1 workload=$2; shift 2
  for context in 131072 65536; do
    check "$model $workload c4 ctx$context" || return
    if python3 workloads.py "$workload" --model "$model" --sessions 4 \
        --workers 4 --context "$context" --tag "c4" "$@"; then
      echo "OK $model $workload ctx$context"; return
    fi
    echo "FAIL $model $workload ctx$context"
  done
}
for model in ${MODELS:-q27 fn}; do
  run "$model" w3 --users 10 --steps 100
  run "$model" w2
done
echo "=== done $(date +%T)"
