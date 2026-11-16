#!/usr/bin/env bash
# Race A: safetensors directory vs .tfold directory through tf-qwen35-cuda stage.
# Produces a JSONL file of per-run measurements.
set -u -o pipefail
BIN="${BIN:-/home/chris/worktrees/tfold/zig-out/bin/tf-qwen35-cuda}"
SFT_DIR="${SFT_DIR:-/home/chris/models/qwen35-2b-orig}"
TFOLD_DIR="${TFOLD_DIR:-/tmp/tfold_dir}"
TOKENS="${TOKENS:-5}"
OUT_JSONL="${OUT_JSONL:-/tmp/race_a.jsonl}"
STAGE_OUT="/tmp/race_a_stage.bin"
AUDIT_SFT="/tmp/race_a_audit_sft.txt"
AUDIT_TFOLD="/tmp/race_a_audit_tfold.txt"
: > "$OUT_JSONL"

one_run() {
    local label="$1"; local dir="$2"; local run_idx="$3"; local cold="$4"
    # ns start
    local t0_ns=$(date +%s%N)
    local tmpout=$(mktemp)
    TF_NORM_AUDIT=1 TF_TFOLD_LOG_STATS=1 TF_PIPE_TOKENS="$TOKENS" \
      /usr/bin/time -f '%e %U %S %M' -o "$tmpout" \
      "$BIN" "$dir" stage "$STAGE_OUT" 2> "/tmp/race_a_stderr_${label}_${run_idx}.txt"
    local rc=$?
    local t1_ns=$(date +%s%N)
    local wall_s=$(awk '{print $1}' "$tmpout")
    local user_s=$(awk '{print $2}' "$tmpout")
    local sys_s=$(awk '{print $3}' "$tmpout")
    local rss_kb=$(awk '{print $4}' "$tmpout")
    rm -f "$tmpout"
    # parse stderr for phase info
    local stderr_file="/tmp/race_a_stderr_${label}_${run_idx}.txt"
    local scenario_ms=$(grep -oE "scenario [0-9]+ ms" "$stderr_file" | head -1 | awk '{print $2}')
    local peak_gib=$(grep -oE "[0-9.]+ GiB peak" "$stderr_file" | head -1 | awk '{print $1}')
    local now_gib=$(grep -oE "[0-9.]+ GiB device now" "$stderr_file" | head -1 | awk '{print $1}')
    local logits_rows=$(grep -oE "[0-9]+ logits rows" "$stderr_file" | head -1 | awk '{print $1}')
    # tfold phase breakdown (present only on .tfold path)
    local fnv_us=$(grep -oE "fnv=[0-9]+us" "$stderr_file" | head -1 | sed 's/fnv=//;s/us//')
    local preamble_us=$(grep -oE "preamble=[0-9]+us" "$stderr_file" | head -1 | sed 's/preamble=//;s/us//')
    local hdr_us=$(grep -oE "header_parse=[0-9]+us" "$stderr_file" | head -1 | sed 's/header_parse=//;s/us//')
    local mmap_us=$(grep -oE "mmap=[0-9]+us" "$stderr_file" | head -1 | sed 's/mmap=//;s/us//')
    # token sequence
    local tokens=$(grep -oE 'tokens [0-9 ]+' "$stderr_file" | head -1 | sed 's/^tokens //')
    # audit lines
    if [[ "$label" == "sft" ]]; then grep tf_norm_audit "$stderr_file" > "$AUDIT_SFT"; fi
    if [[ "$label" == "tfold" ]]; then grep tf_norm_audit "$stderr_file" > "$AUDIT_TFOLD"; fi
    # sha of stage dump
    local dump_sha=$(sha256sum "$STAGE_OUT" | cut -c1-16)
    printf '{"label":"%s","run":%d,"cold":%s,"rc":%d,"wall_s":%s,"user_s":%s,"sys_s":%s,"rss_kb":%s,"scenario_ms":%s,"peak_gib":"%s","now_gib":"%s","logits_rows":"%s","fnv_us":"%s","preamble_us":"%s","header_parse_us":"%s","mmap_us":"%s","dump_sha16":"%s","tokens":"%s"}\n' \
      "$label" "$run_idx" "$cold" "$rc" "$wall_s" "$user_s" "$sys_s" "$rss_kb" "${scenario_ms:-0}" "${peak_gib:-}" "${now_gib:-}" "${logits_rows:-}" "${fnv_us:-0}" "${preamble_us:-0}" "${hdr_us:-0}" "${mmap_us:-0}" "$dump_sha" "$tokens" \
      >> "$OUT_JSONL"
}

echo "=== warmup (both paths once) ==="
one_run sft    "$SFT_DIR"   0 false
one_run tfold  "$TFOLD_DIR" 0 false

echo "=== timed warm runs: 10 per path ==="
for i in $(seq 1 10); do
    one_run sft   "$SFT_DIR"   "$i" false
    one_run tfold "$TFOLD_DIR" "$i" false
done

echo "=== audit line diff ==="
if diff -q "$AUDIT_SFT" "$AUDIT_TFOLD" >/dev/null; then
    echo "AUDIT_IDENTICAL"
else
    echo "AUDIT_DIFFER"; diff "$AUDIT_SFT" "$AUDIT_TFOLD" | head
fi

echo "=== JSONL at $OUT_JSONL ==="
wc -l "$OUT_JSONL"
