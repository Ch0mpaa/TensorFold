#!/usr/bin/env bash
# Stage 6d Gate C: compare three loading paths against identical prompt.
# 1. Native .tfold via patched Checkpoint reader  (NEW in Stage 6d)
# 2. Original safetensors                          (Stage 6c Gate A baseline)
# 3. Materialized safetensors from .tfold          (Stage 6c Gate B baseline)
#
# Each row records: cold-load time (first run), warm-load time (second run),
# peak RSS, inference wall, and output SHA256. Determinism and SHA-match are
# the primary success criteria.
set -euo pipefail

BIN="${BIN:-$HOME/src/tensorfold/zig-out/bin/tf-qwen35-cuda}"
IDS="760,4087,310,2199,11,279,15039,11,321,4156,369"
OUT_DIR="${OUT_DIR:-/tmp/stage6d}"
mkdir -p "$OUT_DIR"

NATIVE_DIR="${NATIVE_DIR:-/tmp/tfold-native}"
SAFETENSORS_DIR="${SAFETENSORS_DIR:-$HOME/models/Qwen3.5-2B-exl3-8.00bpw}"
MATERIALIZED_DIR="${MATERIALIZED_DIR:-/tmp/tfold-expanded}"

run_once() {
    local tag="$1"
    local model_dir="$2"
    local out="$OUT_DIR/out-$tag.bin"
    local log="$OUT_DIR/log-$tag.txt"
    rm -f "$out"
    # Drop page cache so the first run is truly cold. Requires sudo-less /proc?
    # Not available; we measure warm-cache on both the second run of each path.
    sync
    local t0 t1
    t0=$(date +%s.%N)
    /usr/bin/time -v "$BIN" "$model_dir" prompt "$IDS" "$out" > "$log" 2>&1
    t1=$(date +%s.%N)
    local wall peak_kb sha bytes
    wall=$(awk "BEGIN{printf \"%.3f\", $t1 - $t0}")
    peak_kb=$(grep "Maximum resident" "$log" | awk '{print $6}' || echo 0)
    bytes=$(stat -c %s "$out" 2>/dev/null || echo 0)
    sha=$(sha256sum "$out" 2>/dev/null | awk '{print $1}')
    printf '  "%s": {\n    "wall_s": %s,\n    "peak_rss_kb": %s,\n    "bytes": %d,\n    "sha256": "%s"\n  }' \
        "$tag" "$wall" "$peak_kb" "$bytes" "$sha"
}

receipt="$OUT_DIR/gate-c-receipt.json"
{
    printf '{\n  "stage": "6d",\n  "gate": "C",\n  "timestamp_utc": "%s",\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf '  "binary_sha256": "%s",\n' "$(sha256sum "$BIN" | awk '{print $1}')"
    printf '  "prompt_ids": [760, 4087, 310, 2199, 11, 279, 15039, 11, 321, 4156, 369],\n'
    printf '  "measurements": {\n'
    echo "-- native_cold (first touch of model.tfold) --" >&2
    r=$(run_once native_cold "$NATIVE_DIR")
    printf '%s,\n' "$r"
    echo "-- native_warm (page cache hot) --" >&2
    r=$(run_once native_warm "$NATIVE_DIR")
    printf '%s,\n' "$r"
    echo "-- safetensors_cold (original ExLlamaV3 checkpoint) --" >&2
    r=$(run_once safetensors_cold "$SAFETENSORS_DIR")
    printf '%s,\n' "$r"
    echo "-- safetensors_warm --" >&2
    r=$(run_once safetensors_warm "$SAFETENSORS_DIR")
    printf '%s,\n' "$r"
    echo "-- materialized_cold (Stage 6c Gate B path) --" >&2
    r=$(run_once materialized_cold "$MATERIALIZED_DIR")
    printf '%s,\n' "$r"
    echo "-- materialized_warm --" >&2
    r=$(run_once materialized_warm "$MATERIALIZED_DIR")
    printf '%s\n  }' "$r"

    # Parity across the three paths must hold on their last (warm) runs.
    nsha=$(sha256sum "$OUT_DIR/out-native_warm.bin" | awk '{print $1}')
    ssha=$(sha256sum "$OUT_DIR/out-safetensors_warm.bin" | awk '{print $1}')
    msha=$(sha256sum "$OUT_DIR/out-materialized_warm.bin" | awk '{print $1}')
    if [ "$nsha" = "$ssha" ] && [ "$ssha" = "$msha" ]; then match="true"; else match="false"; fi
    printf ',\n  "three_way_sha_match": %s\n}\n' "$match"
} > "$receipt"

cat "$receipt"
