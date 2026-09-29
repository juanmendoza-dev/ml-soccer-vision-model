#!/usr/bin/env bash
# Pack what the CV runs and their reports need on the workstation (09). data/ is gitignored
# and folds.json is already in git. Feature and graph caches are left out: the workstation
# builds its own on the first run.
#
#   scripts/pack_workstation_data.sh [OUT.tar]
#
# Default OUT is data/workstation-data-<date>.tar. Plain tar, since parquet is already
# compressed, and no macOS metadata (._ files, xattrs), so Windows' tar unpacks it cleanly.
# Prints the size and writes OUT.sha256 to check the copy (Get-FileHash on Windows).
set -euo pipefail
cd "$(dirname "$0")/.."

out="${1:-data/workstation-data-$(date +%F).tar}"
list="$(mktemp)"
trap 'rm -f "$list"' EXIT

missing=0
add() {
  if [ -f "$1" ]; then
    echo "$1" >>"$list"
  else
    echo "missing: $1" >&2
    missing=1
  fi
}
for d in data/processed/*/; do
  for f in frames_10hz.parquet objects_10hz.parquet resample_report.json; do add "$d$f"; done
done
for d in data/gamestate/*/; do
  for f in match.parquet events.parquet frames.parquet; do add "$d$f"; done
done
run=data/runs/lgbm-held-2026-09-27
if [ ! -d "$run" ]; then
  echo "missing: $run" >&2
  missing=1
else
  find "$run" -type f | sort >>"$list"
fi
if [ "$missing" -ne 0 ]; then
  echo "not packing: files are missing" >&2
  exit 1
fi

COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -cf "$out" -T "$list"
echo "$(wc -l <"$list" | tr -d ' ') files in $out"
echo "size: $(du -h "$out" | cut -f1) ($(wc -c <"$out" | tr -d ' ') bytes)"
(cd "$(dirname "$out")" && shasum -a 256 "$(basename "$out")") | tee "$out.sha256"
