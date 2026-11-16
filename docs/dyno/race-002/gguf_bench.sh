#!/usr/bin/env bash
# Bench a GGUF checkpoint via llama-server HTTP /completion endpoint.
# Records: cold load time, decode tok/s, VRAM, token sequence for identical prompts.
set -u
LCPP="${LCPP:-/home/chris/.lmstudio/extensions/backends/llama.cpp-linux-arm64-nvidia-cuda13-2.33.0}"
export LD_LIBRARY_PATH="$LCPP:${LD_LIBRARY_PATH:-}"
MODEL="$1"
LABEL="$2"
PORT="${PORT:-18080}"
OUT_JSON="${OUT_JSON:-/tmp/gguf_bench_${LABEL}.json}"
NTOK="${NTOK:-32}"
NGL="${NGL:-99}"   # offload every layer
STDERR="/tmp/gguf_stderr_${LABEL}.txt"

# 1) Start server
"$LCPP/llama-server" -m "$MODEL" --port "$PORT" --host 127.0.0.1 -ngl "$NGL" \
    --ctx-size 2048 --parallel 1 --no-warmup --jinja > "$STDERR" 2>&1 &
PID=$!
# wait for ready
T0=$(date +%s.%N)
for i in $(seq 1 120); do
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then
        break
    fi
    sleep 0.5
done
T1=$(date +%s.%N)
LOAD_S=$(echo "$T1 - $T0" | bc)

# 2) Bench 5 prompts
PROMPTS=(
    "Hello world."
    "The capital of France is"
    "In one sentence, explain photosynthesis."
    "A short poem about rain:"
    "Count to five:"
)
RESULTS="["
for i in "${!PROMPTS[@]}"; do
    P="${PROMPTS[$i]}"
    REQ=$(cat <<EOF
{"prompt": $(printf '%s' "$P" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
 "n_predict": $NTOK, "temperature": 0.0, "top_p": 1.0, "seed": 0}
EOF
)
    TS=$(date +%s.%N)
    R=$(curl -s --max-time 30 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$REQ")
    TE=$(date +%s.%N)
    DT=$(echo "$TE - $TS" | bc)
    TOKENS=$(echo "$R" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tokens_predicted",0))' 2>/dev/null || echo 0)
    TEXT=$(echo "$R" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get("content","")))' 2>/dev/null)
    TPS=$(python3 -c "print(round($TOKENS/$DT,2))" 2>/dev/null || echo 0)
    [ $i -gt 0 ] && RESULTS="$RESULTS,"
    RESULTS="$RESULTS{\"idx\":$i,\"dt_s\":$DT,\"n\":$TOKENS,\"tok_s\":$TPS,\"text\":$TEXT}"
done
RESULTS="$RESULTS]"

# 3) VRAM via nvidia-smi
VRAM_MIB=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
    | awk -F, -v p="$PID" '$1==p {gsub(/ MiB/,"",$2); print $2+0}')

# 4) Shutdown
kill -TERM "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

# 5) Write receipt
cat > "$OUT_JSON" <<EOF
{
  "label": "$LABEL",
  "model_path": "$MODEL",
  "n_predict": $NTOK,
  "ngl": $NGL,
  "cold_load_s": $LOAD_S,
  "vram_mib_observed": ${VRAM_MIB:-0},
  "results": $RESULTS
}
EOF
cat "$OUT_JSON"
