#!/usr/bin/env bash
# Launcher for the codesearch + ollie-rl RL training driver on the VM.
#
# Usage: run-codesearch-ollie.sh <trainer> <name> [extra driver flags...]
#
# Runs the driver from ~/vmsrl-cs with the correct PYTHONPATH, appending all
# output to ~/logs/<name>.log. Meant to be launched inside a detached tmux
# session so it survives the short-lived gcloud ssh connection:
#
#   tmux new-session -d -s <name> "~/run-codesearch-ollie.sh <trainer> <name> ..."
#
# The stable bits (base-url, recipe, dataset) are baked in; per-run knobs
# (--max-datums / --runs / --concurrency / --max-steps / --sample-timeout-s /
# --cloud-run-url) are passed through as extra flags.
set -uo pipefail
ulimit -n 65536

TRAINER="${1:?usage: $0 <trainer> <name> [extra flags]}"
NAME="${2:?usage: $0 <trainer> <name> [extra flags]}"
shift 2

CS_DIR="$HOME/vmsrl-cs"
DATASET="internal_src/example_recipe/codesearch/cognition_gemini4_eval.jsonl"
LOG="$HOME/logs/${NAME}.log"
mkdir -p "$HOME/logs"

cd "$CS_DIR" || { echo "FATAL: no $CS_DIR" | tee -a "$LOG"; exit 2; }

echo "=== $(date -u) launch trainer=$TRAINER name=$NAME extra=[$*] ===" | tee -a "$LOG"

PYTHONPATH="src:internal_src:." .venv/bin/python \
  internal_src/example_recipe/codesearch/codesearch_ollie_train.py \
  --base-url http://127.0.0.1:8000 \
  --trainer "$TRAINER" \
  --name "$NAME" \
  --recipe grpo_4x8 \
  --eval-data-path "$DATASET" \
  "$@" >> "$LOG" 2>&1
rc=$?

echo "=== $(date -u) driver exited rc=$rc ===" | tee -a "$LOG"
exit "$rc"
