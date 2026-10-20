#!/usr/bin/env python3
"""Qualify queue ordering and disconnect cancellation in the native Phone-B server."""

import argparse
import json
import pathlib
import threading
import time

import qualify_qualcomm_native_http as native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process = None; server_pid = None
    report = {"device_serial": native.SERIAL, "server": args.server, "engine": args.engine, "passed": False}
    try:
        process, server_pid, startup = native.start_server(args.server, args.engine)
        report["startup"] = startup
        report["warmup"] = native.stream_request("/v1/completions", {
            "model": "qwen35-4b-exl3-4bpw-local", "prompt": "Warm the local engine.",
            "temperature": 0, "max_tokens": 8, "stream": True,
        })

        results = {}; errors = []
        def run(name, prompt, tokens):
            started = time.monotonic()
            try:
                result = native.stream_request("/v1/completions", {
                    "model": "qwen35-4b-exl3-4bpw-local", "prompt": prompt,
                    "temperature": 0, "max_tokens": tokens, "stream": True,
                })
                result["started_monotonic"] = started
                result["completed_monotonic"] = time.monotonic()
                results[name] = result
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        first = threading.Thread(target=run, args=("first", "Write compact Python code for a queue.", 64))
        second = threading.Thread(target=run, args=("second", "Reply with queued.", 8))
        first.start(); time.sleep(0.05); second.start(); first.join(); second.join()
        if errors: raise RuntimeError("; ".join(errors))
        queue_order = (
            results["second"]["started_monotonic"] < results["first"]["completed_monotonic"] and
            results["second"]["completed_monotonic"] >= results["first"]["completed_monotonic"]
        )
        report["concurrent"] = results
        report["queue_order_preserved"] = queue_order

        worker_before = native.get_json("/health")["worker_pid"]
        report["disconnect_first"] = native.stream_request("/v1/completions", {
            "model": "qwen35-4b-exl3-4bpw-local", "prompt": "Generate a long numbered list with explanations.",
            "temperature": 0, "max_tokens": 256, "stream": True,
        }, disconnect=True)
        recovery = native.stream_request("/v1/completions", {
            "model": "qwen35-4b-exl3-4bpw-local", "prompt": "Reply with recovered.",
            "temperature": 0, "max_tokens": 8, "stream": True,
        })
        health_after = native.get_json("/health")
        metrics = native.get_json("/metrics")
        report["disconnect"] = {
            "worker_pid_before": worker_before,
            "worker_pid_after": health_after["worker_pid"],
            "worker_restarted": worker_before != health_after["worker_pid"],
            "recovery": recovery,
        }
        report["metrics"] = metrics
        report["passed"] = (
            queue_order and report["disconnect"]["worker_restarted"] and
            recovery["usage"]["completion_tokens"] == 8 and
            metrics["cancellations"] >= 1 and metrics["worker_starts"] >= 2
        )
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        native.stop_server(process, server_pid)
        server_name = pathlib.PurePosixPath(args.server).name
        engine_name = pathlib.PurePosixPath(args.engine).name
        report["processes_after"] = native.adb(
            "shell", f"pidof {server_name} {engine_name} 2>/dev/null || true", check=False,
        ).stdout.strip()
        if report["processes_after"]: report["passed"] = False

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "queue_order_preserved": report.get("queue_order_preserved"),
        "worker_restarted": report.get("disconnect", {}).get("worker_restarted"),
        "cancellations": report.get("metrics", {}).get("cancellations"),
        "processes_after": report["processes_after"],
        "error": report.get("error"),
    }))
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
