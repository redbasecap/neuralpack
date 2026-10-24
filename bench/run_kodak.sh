#!/usr/bin/env bash
# Reproduce the Kodak numbers in the README.
#   bench/run_kodak.sh [model] [qualities]
# Downloads the 24 Kodak images (~15 MB) into bench/kodak on first run.
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-mean-scale}"
Q="${2:-1,3,5,8}"
mkdir -p kodak out
for i in $(seq -w 1 24); do
  [ -f "kodak/kodim$i.png" ] || curl -sSL -o "kodak/kodim$i.png" "https://r0k.us/graphics/kodak/kodak/kodim$i.png"
done
npk bench kodak/kodim*.png -m "$MODEL" -q "$Q" --json "out/kodak_$MODEL.json" | tee "out/kodak_$MODEL.txt"
python summarize.py "out/kodak_$MODEL.json"
