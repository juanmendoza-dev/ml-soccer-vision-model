#!/usr/bin/env bash
# usage: queue.sh <file with lines: run_id | cv args>
set -u
cd "/Users/juanmendoza/Desktop/projects/ML Vision Model (Soccerr) -sens"
UV="uv run --extra models --extra dev --extra converters"
BASE=data/runs/lgbm-held-2026-09-27
while IFS='|' read -r id args; do
  id=$(echo "$id" | xargs); [ -z "$id" ] && continue
  if [ -f "data/runs/$id/run.json" ]; then echo "skip $id"; continue; fi
  echo "=== $id: $args ($(date +%H:%M:%S))"
  $UV python -m prediction.cv --model lgbm --horizons h5 --run-id "$id" $args 2>&1 | grep -E "degraded|matches|final|saved" 
  $UV python -m evaluation.compare "data/runs/$id" "$BASE" --out "data/runs/$id/compare.md" >/dev/null 2>&1 && grep -E "^- (PR-AUC|miss|false)" "data/runs/$id/compare.md" | head -3
done < "$1"
echo "queue done $(date +%H:%M:%S)"
