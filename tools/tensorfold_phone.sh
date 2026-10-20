#!/system/bin/sh
set -eu

ROOT=/data/local/tmp/exl3-full-power
MODEL_ROOT=/data/local/tmp/tensorfold-models
STATE=/data/local/tmp/tensorfold-local
SERVER="$ROOT/tensorfold-qualcomm-openai"
ENGINE_4B="$ROOT/tensorfold-qualcomm-exl3-forward-4b"
ENGINE_2B="$ROOT/tensorfold-qualcomm-exl3-forward-2b"
MTP_4B=/data/local/tmp/tensorfold-mtp-poc/mtp.safetensors
TOKENIZER_FALLBACK="$MODEL_ROOT/Qwen3.5-2B-exl3-8.00bpw"
LIB_DIR="$ROOT/lib"
PROFILE="$STATE/profile.conf"
PID_FILE="$STATE/server.pid"
LOG_FILE="$STATE/server.log"
PORT=8082

mkdir -p "$STATE"

alive() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

healthy() {
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1
}

wait_cool() {
    while true; do
        tenths=$(dumpsys battery | grep 'temperature:' | cut -d: -f2 | tr -d ' ')
        [ -z "$tenths" ] && return 0
        [ "$tenths" -le 395 ] && return 0
        echo "Cooling before benchmark: $((tenths / 10)).$((tenths % 10)) C"
        sleep 10
    done
}

safe_path() {
    case "$1" in *[!A-Za-z0-9_./:-]*) return 1;; esac
    return 0
}

write_profile() {
    model_dir=$1
    engine=$2
    mtp=$3
    tokenizer_dir=$4
    model_sha=$5
    cache=$6
    for path in "$model_dir" "$engine" "$mtp" "$tokenizer_dir" "$cache"; do safe_path "$path" || {
        echo "paths with whitespace, quotes, or newlines are not supported" >&2
        return 64
    }; done
    cat >"$PROFILE" <<EOF
MODEL_DIR='$model_dir'
ENGINE='$engine'
MTP='$mtp'
TOKENIZER_DIR='$tokenizer_dir'
MODEL_SHA='$model_sha'
CACHE='$cache'
POWER_MODE='low_balanced'
AMBIGUITY_MARGIN='0.125'
EOF
}

default_profile() {
    model_dir="$MODEL_ROOT/Qwen3.5-4B-exl3-4.00bpw"
    [ -f "$model_dir/model.safetensors" ] || return 1
    [ -f "$model_dir/config.json" ] || return 1
    write_profile "$model_dir" "$ENGINE_4B" "$MTP_4B" "$TOKENIZER_FALLBACK" \
        117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1 \
        /data/local/tmp/tensorfold-exl3-cache-4b-v1
}

load_profile() {
    [ -f "$PROFILE" ] || default_profile || {
        echo "No active model. Run: $0 models; $0 use MODEL_DIR" >&2
        return 1
    }
    . "$PROFILE"
}

models() {
    found=0
    for config in "$MODEL_ROOT"/*/config.json; do
        [ -f "$config" ] || continue
        dir=${config%/config.json}
        found=1
        if meta=$($SERVER inspect "$dir" 2>&1); then
            echo "$dir"
            echo "  $meta"
        else
            echo "$dir"
            echo "  $meta"
        fi
    done
    [ "$found" -eq 1 ] || echo "No model directories with config.json found under $MODEL_ROOT"
}

use_model() {
    [ $# -eq 1 ] || { echo "usage: $0 use MODEL_DIR" >&2; return 64; }
    dir=${1%/}
    [ -f "$dir/config.json" ] || { echo "missing $dir/config.json" >&2; return 65; }
    [ -f "$dir/model.safetensors" ] || { echo "missing $dir/model.safetensors" >&2; return 65; }
    meta=$($SERVER inspect "$dir" 2>&1) || { echo "$meta" >&2; return 65; }
    case "$meta" in
        *'"profile":"qwen3.5-4b"'*) engine=$ENGINE_4B; mtp=$MTP_4B ;;
        *'"profile":"qwen3.5-2b"'*)
            engine=$ENGINE_2B
            mtp=none
            ;;
        *) echo "unsupported model metadata: $meta" >&2; return 65 ;;
    esac
    [ -x "$engine" ] || { echo "missing engine $engine" >&2; return 65; }
    if [ "$mtp" != none ]; then [ -f "$mtp" ] || { echo "missing MTP sidecar $mtp" >&2; return 65; }; fi
    restart_after=0
    if healthy; then restart_after=1; stop_server; fi
    tokenizer_dir=$dir
    [ -f "$tokenizer_dir/tokenizer.json" ] || tokenizer_dir=$TOKENIZER_FALLBACK
    model_sha=$(sha256sum "$dir/model.safetensors" | cut -d' ' -f1)
    model_prefix=$(printf '%s' "$model_sha" | cut -c1-16)
    cache="/data/local/tmp/tensorfold-exl3-cache-$model_prefix"
    if [ "$model_sha" = 117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1 ]; then
        cache=/data/local/tmp/tensorfold-exl3-cache-4b-v1
    fi
    if [ "$model_sha" = a106b1c9d58a768eb87960daa7aff6efd250b0cb7c73edb92155ac9fc653f45f ]; then
        cache=/data/local/tmp/tensorfold-exl3-cache-v1
    fi
    write_profile "$dir" "$engine" "$mtp" "$tokenizer_dir" "$model_sha" "$cache"
    echo "selected $dir ($meta)"
    if [ "$restart_after" -eq 1 ]; then start_server low_balanced 0.125; fi
}

start_server() {
    load_profile
    power_mode=${1:-${POWER_MODE:-low_balanced}}
    ambiguity_margin=${2:-${AMBIGUITY_MARGIN:-0.125}}
    if healthy; then echo "TensorFold is already running"; return 0; fi
    if alive; then kill "$(cat "$PID_FILE")" 2>/dev/null || true; rm -f "$PID_FILE"; fi
    model="$MODEL_DIR/model.safetensors"
    nohup "$SERVER" "$ENGINE" "$model" "$CACHE" "$MODEL_SHA" "$MTP" "$TOKENIZER_DIR" "$LIB_DIR" "$PORT" "$power_mode" "$ambiguity_margin" \
        >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    i=0
    while [ "$i" -lt 90 ]; do
        if healthy; then
            echo "TensorFold local: http://127.0.0.1:$PORT/"
            return 0
        fi
        if ! alive; then cat "$LOG_FILE"; return 1; fi
        i=$((i + 1)); sleep 1
    done
    cat "$LOG_FILE"
    stop_server
    return 1
}

stop_server() {
    if alive; then
        pid=$(cat "$PID_FILE")
        kill "$pid" 2>/dev/null || true
        i=0
        while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 20 ]; do i=$((i + 1)); sleep 1; done
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    echo "TensorFold stopped"
}

status() {
    if healthy; then
        curl -s "http://127.0.0.1:$PORT/health"; echo
        curl -s "http://127.0.0.1:$PORT/metrics"; echo
    else
        echo "TensorFold stopped"
        return 1
    fi
}

open_ui() {
    start_server
    am start -a android.intent.action.VIEW -d "http://127.0.0.1:$PORT/" >/dev/null
}

shortcut() {
    target=/data/data/com.termux/files/home/.shortcuts
    mkdir -p "$target"
    cat >"$target/TensorFold" <<EOF
#!/system/bin/sh
su -c '$ROOT/tensorfold start-ui'
EOF
    chmod 700 "$target/TensorFold"
    echo "Installed Termux:Widget shortcut: TensorFold"
}

doctor() {
    load_profile
    "$SERVER" inspect "$MODEL_DIR"
    sha256sum "$ROOT/tensorfold-qualcomm-exl3-forward-4b-mlperf-corpus-poc" \
        /data/local/tmp/tensorfold-mtp-poc/tensorfold-qualcomm-exl3-forward-4b-live-mtp
    [ -x "$SERVER" ] && [ -x "$ENGINE" ] && [ -f "$MODEL_DIR/model.safetensors" ]
    if [ "$MTP" != none ]; then [ -f "$MTP" ]; fi
    echo "doctor: ok"
}

case "${1:-status}" in
    start) start_server low_balanced 0.125 ;;
    benchmark-start) stop_server; wait_cool; start_server burst none ;;
    start-ui|open) open_ui ;;
    stop) stop_server ;;
    restart) stop_server; start_server low_balanced 0.125 ;;
    status) status ;;
    models) models ;;
    use) shift; use_model "$@" ;;
    shortcut) shortcut ;;
    doctor) doctor ;;
    logs) cat "$LOG_FILE" ;;
    *) echo "usage: $0 start|benchmark-start|start-ui|stop|restart|status|models|use MODEL_DIR|shortcut|doctor|logs" >&2; exit 64 ;;
esac
