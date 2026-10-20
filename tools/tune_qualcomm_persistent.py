#!/usr/bin/env python3
"""Sweep persistent Qualcomm TensorFold MTP recipes on the community chat fixture."""
import argparse, json, os, pathlib, signal, statistics, subprocess, sys, time
import requests

HERE = pathlib.Path(__file__).resolve().parent
ADAPTER = HERE / "qualcomm_persistent_openai.py"
ADB = os.environ.get("TF_ADB", r"D:/projects/phoneproject/.tools/android-sdk/platform-tools/adb.exe")
SERIAL = os.environ.get("TF_SERIAL", "e1db9f1d")
URL = "http://127.0.0.1:18004"
CHAT_PROMPT = "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."
CODE_PROMPT = "Write a short Python function that computes the Fibonacci sequence and explain it."


def battery():
    out = subprocess.run([ADB, "-s", SERIAL, "shell", "dumpsys battery"], capture_output=True, text=True, timeout=10).stdout
    def value(name):
        for line in out.splitlines():
            if line.strip().startswith(name + ":"):
                return int(line.split(":", 1)[1].strip())
        return None
    return {"level": value("level"), "mv": value("voltage"), "temperature_tenths_c": value("temperature")}


def cool(limit=380):
    while True:
        b = battery()
        if b["temperature_tenths_c"] is None or b["temperature_tenths_c"] < limit:
            return b
        print(f"cooling: {b}", flush=True)
        time.sleep(20)


def wait_health(timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(URL + "/health", timeout=1)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise TimeoutError("adapter did not become healthy")


def stream_request(kind, tokens):
    body = {"model": "qwen35-4b-exl3-4bpw-persistent", "max_tokens": tokens,
            "temperature": 0, "seed": 20261019, "stream": True,
            "stream_options": {"include_usage": True}}
    path = "/v1/chat/completions" if kind == "chat" else "/v1/completions"
    if kind == "chat":
        body.update({"messages": [{"role": "user", "content": CHAT_PROMPT}],
                     "chat_template_kwargs": {"enable_thinking": False}})
    else:
        body["prompt"] = CODE_PROMPT
    started = time.perf_counter(); first = last = None; deliveries = 0; final = None
    with requests.post(URL + path, json=body, stream=True, timeout=300) as response:
        response.raise_for_status()
        for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            payload = raw[6:]
            if payload == "[DONE]":
                continue
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"].get("message", "server error"))
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("text")
            if delta is None:
                d = choice.get("delta") or {}
                delta = d.get("content") or d.get("reasoning_content")
            if delta:
                now = time.perf_counter()
                first = now if first is None else first
                last = now; deliveries += 1
            if event.get("usage") is not None:
                final = event
    if final is None or first is None:
        raise RuntimeError("stream lacked usage or text delivery")
    completion = int(final["usage"]["completion_tokens"])
    delivery_s = (last - first) if deliveries >= 2 and last > first else None
    return {"ttft_ms": (first - started) * 1000, "total_ms": (time.perf_counter() - started) * 1000,
            "delivery_tps": (completion - 1) / delivery_s if delivery_s else None,
            "delivery_seconds": delivery_s, "deliveries": deliveries,
            "prompt_tokens": final["usage"]["prompt_tokens"], "completion_tokens": completion,
            "runtime": final.get("runtime") or {}}


def stop_adapter(proc, worker_pid):
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=2)
    if worker_pid:
        subprocess.run([ADB, "-s", SERIAL, "shell", f"kill {worker_pid}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    subprocess.run([ADB, "-s", SERIAL, "forward", "--remove", "tcp:19050"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_recipe(recipe, tokens, repeats, log_dir, fixture):
    before = cool()
    env = os.environ.copy()
    env.update({"TF_MTP_DEPTH": str(recipe["depth"]),
                "TF_MTP_INITIAL_RANK": str(recipe["rank"]),
                "TF_FUSED_LINEAR_MASK": str(recipe.get("mask", 0)),
                "TF_Q8_OUTPUT_END": str(recipe.get("q8_end", 24))})
    if recipe.get("first_margin") is not None:
        env["TF_MTP_FIRST_MARGIN"] = str(recipe["first_margin"])
    if recipe.get("second_margin") is not None:
        env["TF_MTP_SECOND_MARGIN"] = str(recipe["second_margin"])
    name = recipe["name"]
    log_path = log_dir / f"{name}.log"
    worker_pid = None
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(ADAPTER)], env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            health = wait_health(); worker_pid = health.get("worker_pid")
            warmup = stream_request(fixture, tokens)
            samples = [stream_request(fixture, tokens) for _ in range(repeats)]
            result = {"recipe": recipe, "fixture": fixture, "worker_pid": worker_pid, "battery_before": before,
                      "battery_after": battery(), "warmup": warmup, "samples": samples}
            for key in ("delivery_tps", "ttft_ms", "total_ms"):
                result[key + "_median"] = statistics.median(x[key] for x in samples)
            result["token_hashes"] = sorted({x["runtime"].get("token_sha") for x in samples})
            result["accepted_drafts_median"] = statistics.median(x["runtime"].get("accepted_drafts", 0) for x in samples)
            result["rounds_median"] = statistics.median(x["runtime"].get("rounds", 0) for x in samples)
            print(f"{name}: {result['delivery_tps_median']:.3f} tok/s ttft={result['ttft_ms_median']:.1f}ms "
                  f"accepted={result['accepted_drafts_median']} rounds={result['rounds_median']} hashes={result['token_hashes']}", flush=True)
            return result
        finally:
            stop_adapter(proc, worker_pid)
            time.sleep(1)

def recipes(mode):
    if mode == "depth":
        return [{"name": f"d{d}-r{r}", "depth": d, "rank": r} for d in range(1, 7) for r in (1, 2)]
    if mode == "margin":
        base = {"depth": 2, "rank": 1}
        first = [{"name": f"d2-r1-f{x:g}", **base, "first_margin": x} for x in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
        second = [{"name": f"d2-r1-s{x:g}", **base, "second_margin": x} for x in (0.25, 1.0, 4.0)]
        both = [{"name": f"d2-r1-f{x:g}-s{x:g}", **base, "first_margin": x, "second_margin": x} for x in (1.0, 2.0)]
        return first + second + both
    if mode == "final":
        base = {"depth": 2, "rank": 1}
        return ([{"name": "d2-r1-base", **base}] +
                [{"name": f"d2-r1-f{x:g}", **base, "first_margin": x}
                 for x in (0.1, 0.2, 0.25, 0.3, 0.4)])
    if mode == "pair":
        return [{"name": "d2-r1-base", "depth": 2, "rank": 1},
                {"name": "d2-r1-f0.1", "depth": 2, "rank": 1, "first_margin": 0.1}]
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["depth", "margin", "final", "pair"], default="depth")
    ap.add_argument("--fixture", choices=["chat", "code"], default="chat")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out.with_suffix(""); log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for recipe in recipes(args.mode):
            results.append(run_recipe(recipe, args.tokens, args.repeats, log_dir, args.fixture))
            out.write_text(json.dumps({"mode": args.mode, "fixture": args.fixture, "tokens": args.tokens,
                                       "repeats": args.repeats, "results": results}, indent=2), encoding="utf-8")
    finally:
        out.write_text(json.dumps({"mode": args.mode, "fixture": args.fixture, "tokens": args.tokens,
                                   "repeats": args.repeats, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
