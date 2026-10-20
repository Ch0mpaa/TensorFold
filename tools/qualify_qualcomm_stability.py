#!/usr/bin/env python3
"""Run a sustained all-local Qualcomm worker and record thermal/memory stability."""

import argparse
import json
import pathlib
import statistics
import subprocess
import time

ADB = r"D:/projects/phoneproject/.tools/android-sdk/platform-tools/adb.exe"
SERIAL = "e1db9f1d"
BUNDLE = "/data/local/tmp/exl3-full-power"
MODEL = "/data/local/tmp/tensorfold-models/Qwen3.5-4B-exl3-4.00bpw/model.safetensors"
CACHE = "/data/local/tmp/tensorfold-exl3-cache-4b-v1"
MTP = "/data/local/tmp/tensorfold-mtp-poc/mtp.safetensors"
MODEL_SHA = "117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1"
MODE = "serve-q4-full-out-q8-q0-head-adaptive-tail"
PROTECTED = {
    BUNDLE + "/tensorfold-qualcomm-exl3-forward-4b-mlperf-corpus-poc":
        "6917588a7de1b51464822628dce095ad6eb7712db7081b0ec2150b37cb740144",
    "/data/local/tmp/tensorfold-mtp-poc/tensorfold-qualcomm-exl3-forward-4b-live-mtp":
        "99eaeca94fb1c47046ab5e3cbf2c978a05e9eb97ca4342633fcf1c1c81e9a885",
}
FIXTURES = {
    "code": [7734, 264, 2716, 12654, 709, 421, 55288, 279, 76938, 8240, 321, 10033, 424, 13],
    "chat": [248045, 846, 198, 814, 20139, 1204, 5979, 44896, 5533, 264, 21966, 303,
             13983, 6163, 11, 1179, 2873, 264, 2526, 33625, 3010, 13, 248046, 198,
             248045, 74455, 198, 248068, 271, 248069, 271],
}


def adb(*args, timeout=30, check=True):
    result = subprocess.run(
        [ADB, "-s", SERIAL, *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"adb exited {result.returncode}: {result.stdout}\n{result.stderr}")
    return result


def shell(command, timeout=30):
    return adb("shell", command, timeout=timeout).stdout


def temperature_c():
    for line in shell("dumpsys battery").splitlines():
        if line.strip().startswith("temperature:"):
            return int(line.split(":", 1)[1].strip()) / 10.0
    return None


def rss_kib(pid):
    for line in shell(f"cat /proc/{pid}/status 2>/dev/null || true").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def assert_protected_hashes():
    output = shell("sha256sum " + " ".join(PROTECTED))
    actual = {line.split()[1]: line.split()[0] for line in output.splitlines() if line.strip()}
    for path, expected in PROTECTED.items():
        if actual.get(path) != expected:
            raise RuntimeError(f"protected binary changed: {path}: {actual.get(path)}")


def worker_command(binary, steps, power_mode):
    return (
        "su -c '"
        f"cd {BUNDLE}; echo WORKER_PID $$; "
        f"export LD_LIBRARY_PATH={BUNDLE}/lib ADSP_LIBRARY_PATH={BUNDLE}/lib "
        f"GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_HMX_PIPELINE_MIN_ROWS=4 GGML_HEXAGON_POWER_MODE={power_mode} "
        f"TENSORFOLD_MTP_SIDECAR={MTP} TENSORFOLD_MTP_DEPTH=2 "
        "TENSORFOLD_MTP_INITIAL_RANK=1 TENSORFOLD_MTP_FIRST_MARGIN=0.1 "
        "TENSORFOLD_TARGET_AMBIGUITY_MARGIN=0.125 "
        "TENSORFOLD_Q8_OUTPUT_LAYER_END=24 TENSORFOLD_FUSED_LINEAR_LAYER_MASK=0 "
        "TENSORFOLD_BENCH_QUIET=1 TENSORFOLD_STREAM_TOKENS=1 TENSORFOLD_PERSISTENT_WORKER=1; "
        f"exec {binary} htp {MODEL} 0 6 {CACHE} {MODEL_SHA} {steps} 0 {MODE} 1'"
    )


def read_event(process):
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"worker closed with status {process.poll()}")
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


def start_worker(binary, steps, power_mode):
    process = subprocess.Popen(
        [ADB, "-s", SERIAL, "shell", worker_command(binary, steps, power_mode)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    pid = None
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("worker exited before ready")
        line = line.strip()
        if line.startswith("WORKER_PID "):
            pid = int(line.split()[1])
        elif line.startswith("{"):
            event = json.loads(line)
            if event.get("tensorfold") == "qualcomm-exl3-worker-ready":
                if pid is None:
                    raise RuntimeError("worker did not report its pid")
                return process, pid


def run_request(process, request_id, tokens, steps):
    process.stdin.write(f"{request_id} {steps} {','.join(map(str, tokens))}\n")
    process.stdin.flush()
    started = time.monotonic()
    generated = []
    prefill = summary = done = None
    while done is None:
        event = read_event(process)
        kind = event.get("tensorfold")
        if kind == "qualcomm-exl3-speculative-chunk":
            generated.extend(int(token) for token in event["tokens"][:int(event["count"])])
        elif kind == "qualcomm-exl3-prefill":
            prefill = event
        elif kind == "qualcomm-exl3-speculative-summary":
            summary = event
        elif kind == "qualcomm-exl3-worker-done":
            if int(event.get("id", -1)) != request_id:
                raise RuntimeError("worker response id mismatch")
            done = event
    if summary is None or len(generated) != steps:
        raise RuntimeError(f"incomplete request {request_id}: {len(generated)}/{steps}")
    return {
        "request_id": request_id,
        "wall_seconds": time.monotonic() - started,
        "token_hash": summary["token_hash"],
        "serial_fallbacks": int(summary.get("serial_fallbacks", 0)),
        "decode_seconds": float(summary["speculative_elapsed_ms"]) / 1000.0,
        "tokens_per_second": steps * 1000.0 / float(summary["speculative_elapsed_ms"]),
        "prefill_ms": None if prefill is None else float(prefill["elapsed_ms"]),
        "prompt_cache_hit": False if prefill is None else bool(prefill.get("cache_hit", False)),
        "resident_weight_uploads": int(done["resident_weight_uploads"]),
        "mtp_projection_uploads": int(done["mtp_projection_uploads"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-seconds", type=int, default=1800)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--sample-seconds", type=int, default=30)
    parser.add_argument("--thermal-pause-c", type=float, default=42.0)
    parser.add_argument("--thermal-resume-c", type=float, default=39.5)
    parser.add_argument("--thermal-limit-c", type=float, default=45.0)
    parser.add_argument("--power-mode", default="low_balanced")
    parser.add_argument("--start-max-c", type=float, default=39.5)
    args = parser.parse_args()

    assert_protected_hashes()
    process = None
    pid = None
    requests = []
    samples = []
    prestart = time.monotonic()
    while True:
        start_temperature = temperature_c()
        if start_temperature is None or start_temperature <= args.start_max_c:
            break
        time.sleep(10)
    prestart_cooldown_seconds = time.monotonic() - prestart
    started = time.monotonic()
    next_sample = started
    error = None
    cooldown_seconds = 0.0
    try:
        process, pid = start_worker(args.binary, args.tokens, args.power_mode)
        while time.monotonic() - started < args.duration_seconds:
            fixture = "code" if len(requests) % 2 == 0 else "chat"
            result = run_request(process, len(requests) + 1, FIXTURES[fixture], args.tokens)
            result["fixture"] = fixture
            result["elapsed_seconds"] = time.monotonic() - started
            requests.append(result)
            if time.monotonic() >= next_sample:
                sample = {
                    "elapsed_seconds": time.monotonic() - started,
                    "temperature_c": temperature_c(),
                    "rss_kib": rss_kib(pid),
                    "state": "active",
                }
                samples.append(sample)
                next_sample = time.monotonic() + args.sample_seconds
                if sample["temperature_c"] is not None and sample["temperature_c"] >= args.thermal_limit_c:
                    raise RuntimeError(f"thermal limit reached: {sample['temperature_c']} C")
                if sample["temperature_c"] is not None and sample["temperature_c"] >= args.thermal_pause_c:
                    paused = time.monotonic()
                    while time.monotonic() - started < args.duration_seconds:
                        time.sleep(10)
                        cooling = {
                            "elapsed_seconds": time.monotonic() - started,
                            "temperature_c": temperature_c(),
                            "rss_kib": rss_kib(pid),
                            "state": "cooldown",
                        }
                        samples.append(cooling)
                        if cooling["temperature_c"] is not None and cooling["temperature_c"] >= args.thermal_limit_c:
                            raise RuntimeError(f"thermal limit reached: {cooling['temperature_c']} C")
                        if cooling["temperature_c"] is None or cooling["temperature_c"] <= args.thermal_resume_c:
                            break
                    cooldown_seconds += time.monotonic() - paused
                    next_sample = time.monotonic() + args.sample_seconds
    except Exception as exc:
        error = str(exc)
    finally:
        if process is not None:
            try:
                if process.stdin: process.stdin.close()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                if pid is not None:
                    adb("shell", f"su -c 'kill -9 {pid} 2>/dev/null || true'", check=False)
                process.kill()
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired: pass

    elapsed = time.monotonic() - started
    assert_protected_hashes()
    hashes = {
        fixture: sorted({request["token_hash"] for request in requests if request["fixture"] == fixture})
        for fixture in FIXTURES
    }
    warm_requests = requests[2:]
    rss_values = [sample["rss_kib"] for sample in samples[1:] if sample["rss_kib"] is not None]
    stable_memory = not rss_values or max(rss_values) - min(rss_values) <= 128 * 1024
    no_warm_uploads = all(
        request["resident_weight_uploads"] == 0 and request["mtp_projection_uploads"] == 0
        for request in warm_requests
    )
    passed = (
        error is None and elapsed >= args.duration_seconds and len(requests) >= 2 and
        all(len(values) == 1 for values in hashes.values()) and stable_memory and no_warm_uploads
    )
    report = {
        "device_serial": SERIAL,
        "binary": args.binary,
        "power_mode": args.power_mode,
        "duration_target_seconds": args.duration_seconds,
        "elapsed_seconds": elapsed,
        "tokens_per_request": args.tokens,
        "request_count": len(requests),
        "start_max_c": args.start_max_c,
        "start_temperature_c": start_temperature,
        "prestart_cooldown_seconds": prestart_cooldown_seconds,
        "thermal_pause_c": args.thermal_pause_c,
        "thermal_resume_c": args.thermal_resume_c,
        "thermal_limit_c": args.thermal_limit_c,
        "cooldown_seconds": cooldown_seconds,
        "active_request_seconds": sum(request["wall_seconds"] for request in requests),
        "request_duty_cycle": (sum(request["wall_seconds"] for request in requests) / elapsed) if elapsed > 0 else 0,
        "temperature_c_min": min((sample["temperature_c"] for sample in samples if sample["temperature_c"] is not None), default=None),
        "temperature_c_max": max((sample["temperature_c"] for sample in samples if sample["temperature_c"] is not None), default=None),
        "rss_kib_min_after_warmup": min(rss_values, default=None),
        "rss_kib_max_after_warmup": max(rss_values, default=None),
        "stable_memory": stable_memory,
        "no_weight_or_mtp_reuploads_after_warmup": no_warm_uploads,
        "token_hashes": hashes,
        "median_tokens_per_second": statistics.median([request["tokens_per_second"] for request in requests]) if requests else None,
        "error": error,
        "passed": passed,
        "samples": samples,
        "requests": requests,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "passed", "elapsed_seconds", "request_count", "temperature_c_min", "temperature_c_max",
        "rss_kib_min_after_warmup", "rss_kib_max_after_warmup", "median_tokens_per_second", "error",
    )}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
