#!/system/bin/sh
set -eu

BUNDLE=/data/local/tmp/exl3-full-power
TERMUX=/data/data/com.termux/files/usr
TERMUX_HOME=/data/data/com.termux/files/home
HTTP_PORT=8082
HTTP_PID=/data/local/tmp/tf-local-http.pid
HTTP_LOG=/data/local/tmp/tf-local-http.log
SERVER=$BUNDLE/tensorfold-local-http-v1.py

alive() {
    [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null
}

healthy() {
    curl -sf -m 1 "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

wait_health() {
    port=$1
    i=0
    while [ "$i" -lt 60 ]; do
        if healthy "$port"; then return 0; fi
        i=$((i + 1))
        sleep 1
    done
    return 1
}


start_http() {
    if healthy "$HTTP_PORT"; then return 0; fi
    HOME=$TERMUX_HOME PREFIX=$TERMUX TMPDIR=$TERMUX/tmp nohup "$TERMUX/bin/python" "$SERVER" \
        >"$HTTP_LOG" 2>&1 &
    echo $! >"$HTTP_PID"
    wait_health "$HTTP_PORT" || { cat "$HTTP_LOG"; return 1; }
}

stop_one() {
    file=$1
    if alive "$file"; then kill "$(cat "$file")" 2>/dev/null || true; fi
    rm -f "$file"
}

case "${1:-status}" in
    start)
        start_http
        echo "TensorFold local API: http://0.0.0.0:$HTTP_PORT/v1"
        ;;
    stop)
        stop_one "$HTTP_PID"
        sleep 1
        for pid in $(pidof tensorfold-qualcomm-exl3-forward-4b-persistent-http-v1 2>/dev/null || true); do kill "$pid" 2>/dev/null || true; done
        echo "stopped"
        ;;
    status)
        echo "tensorfold=$(curl -s -m 1 -o /dev/null -w '%{http_code}' http://127.0.0.1:$HTTP_PORT/health 2>/dev/null || true)"
        curl -s -m 1 http://127.0.0.1:$HTTP_PORT/health 2>/dev/null || true
        echo
        ;;
    *)
        echo "usage: $0 start|stop|status" >&2
        exit 64
        ;;
esac
