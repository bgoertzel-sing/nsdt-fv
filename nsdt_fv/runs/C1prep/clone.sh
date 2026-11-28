#!/bin/bash
# C1prep step 1: bare, blob-size-limited clones of the 23 VeriSoftBench repositories (URLs from bench/vsb/repos.json)
# into a scratch directory OUTSIDE this repository, then the source-level history scan. github.com only; no build.
#   nsdt_fv/runs/C1prep/clone.sh /path/to/scratch
set -euo pipefail
SCRATCH=${1:?scratch dir}
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$SCRATCH/clones"
python3 - "$HERE/nsdt_fv/bench/vsb/repos.json" > "$SCRATCH/urls.txt" <<'EOF'
import json, sys
for k, v in sorted(json.load(open(sys.argv[1]))["repos"].items()):
    print(k, v["url"])
EOF
while read -r name url; do
  [ -d "$SCRATCH/clones/$name.git" ] && continue
  git clone --bare --filter=blob:limit=2m --quiet "$url" "$SCRATCH/clones/$name.git"
done < "$SCRATCH/urls.txt"
cd "$HERE"
python3 -m nsdt_fv.bench.vsb_history --clones "$SCRATCH/clones" --out nsdt_fv/runs/C1prep/history
python3 -m nsdt_fv.eval.c1prep t3 --real "$SCRATCH/clones"
python3 -m nsdt_fv.eval.c1prep tables
