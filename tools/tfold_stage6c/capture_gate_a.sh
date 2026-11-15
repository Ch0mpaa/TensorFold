#!/usr/bin/env bash
# Stage 6c Gate A capture.
# Runs tf-qwen35-cuda prompt twice against the patched binary, records both
# outputs, verifies byte-identical determinism, and writes a JSON receipt.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.5-2B-exl3-8.00bpw}"
BIN="${BIN:-$HOME/src/tensorfold/zig-out/bin/tf-qwen35-cuda}"
IDS="760,4087,310,2199,11,279,15039,11,321,4156,369"
PROMPT='The answer to life, the universe, and everything is'
OUT_DIR="${OUT_DIR:-/tmp/stage6c}"
RECEIPT="${RECEIPT:-$OUT_DIR/receipt.json}"

mkdir -p "$OUT_DIR"

run_once() {
    local tag="$1"
    local out="$OUT_DIR/out-$tag.bin"
    local log="$OUT_DIR/log-$tag.txt"
    rm -f "$out"
    local t0=$(date +%s.%N)
    "$BIN" "$MODEL_DIR" prompt "$IDS" "$out" > "$log" 2>&1
    local t1=$(date +%s.%N)
    local wall=$(awk "BEGIN{printf \"%.3f\", $t1 - $t0}")
    local bytes=$(stat -c %s "$out")
    local sha=$(sha256sum "$out" | awk '{print $1}')
    printf '  "%s": {\n    "wall_s": %s,\n    "bytes": %d,\n    "sha256": "%s"\n  }' "$tag" "$wall" "$bytes" "$sha"
}

exec > >(tee "$OUT_DIR/capture.log") 2>&1
echo "== Stage 6c Gate A =="
date -u
echo "model_dir : $MODEL_DIR"
echo "binary    : $BIN"
echo "prompt    : $PROMPT"
echo "ids       : $IDS"

printf '{\n  "stage": "6c",\n  "gate": "A",\n  "timestamp_utc": "%s",\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" > "$RECEIPT"
printf '  "model_dir": "%s",\n  "model_checkpoint_sha": "a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f",\n' "$MODEL_DIR" >> "$RECEIPT"
printf '  "binary": "%s",\n  "binary_sha256": "%s",\n' "$BIN" "$(sha256sum "$BIN" | awk '{print $1}')" >> "$RECEIPT"
printf '  "prompt_text": "%s",\n  "prompt_ids": [760, 4087, 310, 2199, 11, 279, 15039, 11, 321, 4156, 369],\n' "$PROMPT" >> "$RECEIPT"
printf '  "runs": {\n' >> "$RECEIPT"

echo "-- run 1 --"
r1="$(run_once run1)"
printf '%s,\n' "$r1" >> "$RECEIPT"
echo "-- run 2 --"
r2="$(run_once run2)"
printf '%s\n  },\n' "$r2" >> "$RECEIPT"

sha1=$(sha256sum "$OUT_DIR/out-run1.bin" | awk '{print $1}')
sha2=$(sha256sum "$OUT_DIR/out-run2.bin" | awk '{print $1}')
if [ "$sha1" = "$sha2" ]; then det=true; else det=false; fi
printf '  "deterministic_match": %s\n}\n' "$det" >> "$RECEIPT"

echo "== Receipt =="
cat "$RECEIPT"
