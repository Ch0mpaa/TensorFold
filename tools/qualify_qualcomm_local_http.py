#!/usr/bin/env python3
"""Qualify Phone-B-local HTTP queueing, concurrency, and disconnect cancellation."""

import argparse
import http.client
import json
import pathlib
import socket
import subprocess
import threading
import time
import urllib.request

ADB = r"D:/projects/phoneproject/.tools/android-sdk/platform-tools/adb.exe"
SERIAL = "e1db9f1d"
BUNDLE = "/data/local/tmp/exl3-full-power"
TERMUX = "/data/data/com.termux/files/usr"
TERMUX_HOME = "/data/data/com.termux/files/home"
DEVICE_PORT = 8083
HOST_PORT = 18083
SERVER = BUNDLE + "/tensorfold-local-http-qualification.py"


def adb(*args, timeout=30, check=True):
    result = subprocess.run(
        [ADB, "-s", SERIAL, *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"adb exited {result.returncode}: {result.stdout}\n{result.stderr}")
    return result


def health():
    with urllib.request.urlopen(f"http://127.0.0.1:{HOST_PORT}/health", timeout=3) as response:
        return json.load(response)


def start_server(binary):
    adb("forward", "--remove", f"tcp:{HOST_PORT}", check=False)
    adb("forward", f"tcp:{HOST_PORT}", f"tcp:{DEVICE_PORT}")
    command = (
        "su -c '"
        f"cd {BUNDLE}; echo SERVER_PID $$; "
        f"export HOME={TERMUX_HOME} PREFIX={TERMUX} TMPDIR={TERMUX}/tmp "
        f"TF_HTTP_HOST=127.0.0.1 TF_HTTP_PORT={DEVICE_PORT} TF_BINARY={binary}; "
        f"exec {TERMUX}/bin/python {SERVER}'"
    )
    process = subprocess.Popen(
        [ADB, "-s", SERIAL, "shell", command], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    server_pid = None
    deadline = time.monotonic() + 60
    startup = []
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"local server exited before ready: {startup[-10:]}")
        line = line.strip()
        startup.append(line)
        if line.startswith("SERVER_PID "):
            server_pid = int(line.split()[1])
        if "TensorFold local API on" in line:
            if server_pid is None:
                raise RuntimeError("local server did not report pid")
            state = health()
            if state.get("status") != "ok":
                raise RuntimeError(f"local server unhealthy: {state}")
            return process, server_pid, startup
    raise TimeoutError("local server startup timed out")


def stop_server(process, server_pid):
    if server_pid is not None:
        adb("shell", f"su -c 'kill -2 {server_pid} 2>/dev/null || true'", check=False)
    if process is not None:
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if server_pid is not None:
                adb("shell", f"su -c 'kill -9 {server_pid} 2>/dev/null || true'", check=False)
            process.kill()
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired: pass
    adb("forward", "--remove", f"tcp:{HOST_PORT}", check=False)


def request(prompt, max_tokens):
    body = json.dumps({
        "model": "qwen35-4b-exl3-4bpw-local", "prompt": prompt,
        "temperature": 0, "max_tokens": max_tokens, "stream": True,
    }).encode()
    connection = http.client.HTTPConnection("127.0.0.1", HOST_PORT, timeout=120)
    started = time.monotonic()
    first_event = None
    final_event = None
    try:
        connection.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read().decode(errors='replace')}")
        while True:
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if first_event is None:
                first_event = time.monotonic() - started
            if event.get("usage"):
                final_event = event
    finally:
        connection.close()
    if final_event is None:
        raise RuntimeError("stream ended without final usage")
    return {
        "started_monotonic": started,
        "completed_monotonic": time.monotonic(),
        "elapsed_seconds": time.monotonic() - started,
        "first_event_seconds": first_event,
        "runtime": final_event["runtime"],
        "usage": final_event["usage"],
    }


def disconnected_request(prompt):
    body = json.dumps({
        "model": "qwen35-4b-exl3-4bpw-local", "prompt": prompt,
        "temperature": 0, "max_tokens": 256, "stream": True,
    }).encode()
    connection = http.client.HTTPConnection("127.0.0.1", HOST_PORT, timeout=30)
    connection.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"disconnect request HTTP {response.status}")
    first = None
    while first is None:
        line = response.readline()
        if not line:
            raise RuntimeError("disconnect stream ended before first event")
        if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
            first = json.loads(line[6:])
    if connection.sock is not None:
        try: connection.sock.shutdown(socket.SHUT_RDWR)
        except OSError: pass
    connection.close()
    return first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process = None
    server_pid = None
    report = {"device_serial": SERIAL, "binary": args.binary, "passed": False}
    try:
        process, server_pid, startup = start_server(args.binary)
        report["startup"] = startup
        report["warmup"] = request("Return the word warmup repeatedly.", 8)

        results = {}
        errors = []
        def run_named(name, prompt, tokens):
            try: results[name] = request(prompt, tokens)
            except Exception as exc: errors.append(f"{name}: {exc}")

        first = threading.Thread(target=run_named, args=("first", "Write compact Python code for a queue.", 64))
        second = threading.Thread(target=run_named, args=("second", "Reply with queued.", 8))
        first.start(); time.sleep(0.05); second.start(); first.join(); second.join()
        if errors:
            raise RuntimeError("; ".join(errors))
        report["concurrent"] = results
        queued = (
            results["second"]["started_monotonic"] < results["first"]["completed_monotonic"] and
            results["second"]["completed_monotonic"] >= results["first"]["completed_monotonic"]
        )
        report["queue_order_preserved"] = queued

        worker_before = health()["worker_pid"]
        report["disconnect_first_event"] = disconnected_request("Generate a long numbered list with explanations.")
        recovery = request("Reply with recovered.", 8)
        worker_after = health()["worker_pid"]
        report["disconnect"] = {
            "worker_pid_before": worker_before,
            "worker_pid_after": worker_after,
            "worker_restarted": worker_before != worker_after,
            "recovery": recovery,
        }
        report["passed"] = queued and worker_before != worker_after and recovery["usage"]["completion_tokens"] == 8
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        stop_server(process, server_pid)
        report["server_processes_after"] = adb(
            "shell", "pidof tensorfold-qualcomm-exl3-forward-4b-exactguard-v2 2>/dev/null || true",
            check=False,
        ).stdout.strip()
        if report["server_processes_after"]:
            report["passed"] = False

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "queue_order_preserved": report.get("queue_order_preserved"),
        "worker_restarted": report.get("disconnect", {}).get("worker_restarted"),
        "server_processes_after": report["server_processes_after"],
        "error": report.get("error"),
    }))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
