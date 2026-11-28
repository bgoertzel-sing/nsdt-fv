#!/usr/bin/env bash
# S5: fetch VeriSoftBench's task file and repos.json at a pinned commit and verify their sha256.
# Network: github.com (ls-tree of the pinned commit) and raw.githubusercontent.com (files) only.
# Usage: nsdt_fv/bench/vsb_fetch.sh [out_dir]   (default: nsdt_fv/bench/vsb; files are committed there)
set -euo pipefail
COMMIT=8f5e8b36ab5fbdec6fd8d8f4ee0580d29647097a      # utopia-group/VeriSoftBench main on 2026-11-28 00:41 UTC
BASE=https://raw.githubusercontent.com/utopia-group/VeriSoftBench/$COMMIT
OUT=${1:-$(dirname "$0")/vsb}
mkdir -p "$OUT"
curl -fsS -o "$OUT/verisoftbench.jsonl" "$BASE/data/verisoftbench.jsonl"
curl -fsS -o "$OUT/repos.json" "$BASE/repos.json"
curl -fsS -o "$OUT/LICENSE" "$BASE/LICENSE"
# file listing of the pinned commit (names only; no blobs): used to count the two large repositories'
# pre-generated full-context prompts (prompts/full_limited/<repo>/)
tmp=$(mktemp -d)
git clone -q --filter=blob:none --no-checkout https://github.com/utopia-group/VeriSoftBench "$tmp/vsb"
git -C "$tmp/vsb" ls-tree -r --name-only "$COMMIT" > "$OUT/tree.txt"
rm -rf "$tmp"
cd "$OUT" && sha256sum -c <<EOF
ea584250021fbfaddc060834be2e05b9b2fae4f720c770a9932c71d039a50961  verisoftbench.jsonl
c9cc7cb9650c482d81ee78d7f704c61d0d6073418064d11c55674938f35760de  repos.json
EOF
