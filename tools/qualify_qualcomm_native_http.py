#!/usr/bin/env python3
"""Qualify the Phone-B-native Zig tokenizer and OpenAI HTTP control plane."""

import argparse
import http.client
import json
import pathlib
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qwen35_tokenizer import Qwen35Tokenizer

ADB = r"D:/projects/phoneproject/.tools/android-sdk/platform-tools/adb.exe"
SERIAL = "e1db9f1d"
BUNDLE = "/data/local/tmp/exl3-full-power"
MODEL = "/data/local/tmp/tensorfold-models/Qwen3.5-4B-exl3-4.00bpw/model.safetensors"
CACHE = "/data/local/tmp/tensorfold-exl3-cache-4b-v1"
MTP = "/data/local/tmp/tensorfold-mtp-poc/mtp.safetensors"
MODEL_SHA = "117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1"
TOKENIZER_DEVICE = "/data/local/tmp/tensorfold-models/Qwen3.5-2B-exl3-8.00bpw"
LIB_DIR = BUNDLE + "/lib"
DEVICE_PORT = 8084
HOST_PORT = 18084


def adb(*args, timeout=30, check=True):
    result = subprocess.run(
        [ADB, "-s", SERIAL, *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"adb exited {result.returncode}: {result.stdout}\n{result.stderr}")
    return result


def get_json(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{HOST_PORT}{path}", timeout=10) as response:
        return json.load(response)


def post_json(path, value, timeout=30):
    body = json.dumps(value, ensure_ascii=False).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", HOST_PORT, timeout=timeout)
    try:
        connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = response.read()
        decoded = json.loads(payload)
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {decoded}")
        return decoded
    finally:
        connection.close()


def stream_request(path, value, disconnect=False):
    body = json.dumps(value, ensure_ascii=False).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", HOST_PORT, timeout=180)
    started = time.monotonic()
    connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read().decode(errors='replace')}")
    first = None
    final = None
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
        if first is None:
            first = time.monotonic() - started
            if disconnect:
                if connection.sock is not None:
                    try: connection.sock.shutdown(socket.SHUT_RDWR)
                    except OSError: pass
                connection.close()
                return {"first_event_seconds": first, "event": event}
        if event.get("usage"):
            final = event
    connection.close()
    if final is None:
        raise RuntimeError("stream ended without final usage")
    return {
        "elapsed_seconds": time.monotonic() - started,
        "first_event_seconds": first,
        "usage": final["usage"],
        "runtime": final["runtime"],
    }


def start_server(server, engine):
    adb("forward", "--remove", f"tcp:{HOST_PORT}", check=False)
    adb("forward", f"tcp:{HOST_PORT}", f"tcp:{DEVICE_PORT}")
    command = (
        f"su -c 'cd {BUNDLE}; echo SERVER_PID $$; exec {server} {engine} {MODEL} {CACHE} "
        f"{MODEL_SHA} {MTP} {TOKENIZER_DEVICE} {LIB_DIR} {DEVICE_PORT}'"
    )
    process = subprocess.Popen(
        [ADB, "-s", SERIAL, "shell", command], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    server_pid = None
    startup = []
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"native server exited before ready: {startup[-20:]}")
        line = line.strip(); startup.append(line)
        if line.startswith("SERVER_PID "):
            server_pid = int(line.split()[1])
        if line.startswith("{"):
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            if event.get("tensorfold") == "qualcomm-native-http-ready":
                if server_pid is None: raise RuntimeError("native server did not report pid")
                return process, server_pid, startup
    raise TimeoutError("native server startup timed out")


def stop_server(process, server_pid):
    if server_pid is not None:
        adb("shell", f"su -c 'kill -15 {server_pid} 2>/dev/null || true'", check=False)
    if process is not None:
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if server_pid is not None:
                adb("shell", f"su -c 'kill -9 {server_pid} 2>/dev/null || true'", check=False)
            process.kill()
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired: pass
    adb("forward", "--remove", f"tcp:{HOST_PORT}", check=False)


def cases():
    fixed = [
        "", "hello", " hello", "hello\nworld", "<|im_start|>user\nhello<|im_end|>\n",
        "café", "Cafe\u0301", "中文测试", "日本語", "한국어", "🙂🚀", "a\u200db", "\t  \r\n",
        "def f(x):\n    return x * 2\n", "'re 'VE 'll 'S", "नमस्ते", "مرحبا", "Привет",
    ]
    rng = random.Random(0x54464F4C44)
    alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?\n\t") + list("éΩ中🙂🚀ßøЖあ한")
    while len(fixed) < 511:
        fixed.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 180))))
    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()

    reference = Qwen35Tokenizer(args.tokenizer)
    process = None; server_pid = None
    report = {"device_serial": SERIAL, "server": args.server, "engine": args.engine, "passed": False}
    try:
        process, server_pid, startup = start_server(args.server, args.engine)
        report["startup"] = startup
        report["health_before"] = get_json("/health")
        report["manifest"] = get_json("/v1/benchmark/manifest")

        mismatches = []
        encoded_cases = []
        for index, text in enumerate(cases()):
            expected = reference.encode(text, parse_special=True)
            actual = post_json("/v1/tokenize", {"prompt": text})["tokens"]
            if actual != expected:
                mismatches.append({"index": index, "text": text, "expected": expected, "actual": actual})
            encoded_cases.append((text, expected))
        detokenize_mismatches = []
        for index, (text, token_ids) in enumerate(encoded_cases[:201]):
            expected = reference.decode(token_ids)
            actual = post_json("/v1/detokenize", {"tokens": token_ids})["content"]
            if actual != expected:
                detokenize_mismatches.append({"index": index, "expected": expected, "actual": actual})

        chat_cases = [
            ([{"role": "user", "content": "Hello"}], False),
            ([{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Why?"}], False),
            ([{"role": "user", "content": "Think."}], True),
            ([{"role": "user", "content": "Use tool"}, {"role": "assistant", "content": "Calling"}, {"role": "tool", "content": "done"}], False),
        ]
        chat_mismatches = []
        for index, (messages, thinking) in enumerate(chat_cases):
            expected = reference.encode(reference.chat_template(messages, thinking), parse_special=True)
            actual = post_json("/v1/tokenize", {
                "messages": messages, "chat_template_kwargs": {"enable_thinking": thinking},
            })["tokens"]
            if actual != expected:
                chat_mismatches.append({"index": index, "expected": expected, "actual": actual})

        report["tokenizer"] = {
            "encode_cases": len(encoded_cases), "encode_mismatches": mismatches,
            "decode_cases": 201, "decode_mismatches": detokenize_mismatches,
            "chat_cases": len(chat_cases), "chat_mismatches": chat_mismatches,
        }
        if not args.tokenizer_only:
            report["completion"] = stream_request("/v1/completions", {
                "model": "qwen35-4b-exl3-4bpw-local", "prompt": "Write a tiny Python function.",
                "temperature": 0, "max_tokens": 16, "stream": True,
            })
            report["chat"] = stream_request("/v1/chat/completions", {
                "model": "qwen35-4b-exl3-4bpw-local", "messages": [{"role": "user", "content": "Say hello."}],
                "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0,
                "max_tokens": 16, "stream": True,
            })
        report["metrics"] = get_json("/metrics")
        report["health_after"] = get_json("/health")
        generation_ok = args.tokenizer_only or (
            report["completion"]["usage"]["completion_tokens"] == 16 and
            report["chat"]["usage"]["completion_tokens"] == 16
        )
        report["passed"] = (
            not mismatches and not detokenize_mismatches and not chat_mismatches and
            generation_ok and
            report["manifest"]["runtime"]["python_version"] == "none" and
            report["manifest"]["runtime"]["backend"] == "qualcomm"
        )
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        stop_server(process, server_pid)
        server_name = pathlib.PurePosixPath(args.server).name
        engine_name = pathlib.PurePosixPath(args.engine).name
        report["processes_after"] = adb(
            "shell", f"pidof {server_name} {engine_name} 2>/dev/null || true",
            check=False,
        ).stdout.strip()
        if report["processes_after"]:
            report["passed"] = False

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "encode_mismatches": len(report.get("tokenizer", {}).get("encode_mismatches", [])),
        "decode_mismatches": len(report.get("tokenizer", {}).get("decode_mismatches", [])),
        "chat_mismatches": len(report.get("tokenizer", {}).get("chat_mismatches", [])),
        "processes_after": report["processes_after"],
        "error": report.get("error"),
    }))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
