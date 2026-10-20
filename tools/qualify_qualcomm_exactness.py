#!/usr/bin/env python3
"""Prove Qualcomm speculative output matches the current serial target on-device."""

import argparse
import json
import pathlib
import random
import subprocess
import tempfile

ADB = r"D:/projects/phoneproject/.tools/android-sdk/platform-tools/adb.exe"
SERIAL = "e1db9f1d"
BUNDLE = "/data/local/tmp/exl3-full-power"
BINARY = BUNDLE + "/tensorfold-qualcomm-exl3-forward-4b-policy-dump-v1"
MODEL = "/data/local/tmp/tensorfold-models/Qwen3.5-4B-exl3-4.00bpw/model.safetensors"
CACHE = "/data/local/tmp/tensorfold-exl3-cache-4b-v1"
MTP = "/data/local/tmp/tensorfold-mtp-poc/mtp.safetensors"
MODEL_SHA = "117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1"
MODE = "serve-q4-full-out-q8-q0-head-adaptive-tail"
REMOTE_CORPUS = BUNDLE + "/qualification-exactness-corpus.json"
PROTECTED = {
    BUNDLE + "/tensorfold-qualcomm-exl3-forward-4b-mlperf-corpus-poc":
        "6917588a7de1b51464822628dce095ad6eb7712db7081b0ec2150b37cb740144",
    "/data/local/tmp/tensorfold-mtp-poc/tensorfold-qualcomm-exl3-forward-4b-live-mtp":
        "99eaeca94fb1c47046ab5e3cbf2c978a05e9eb97ca4342633fcf1c1c81e9a885",
}
SELECTED_IFEVAL = [
    "ifeval-007", "ifeval-008", "ifeval-019", "ifeval-003", "ifeval-028",
    "ifeval-016", "ifeval-023", "ifeval-017", "ifeval-022", "ifeval-002",
    "ifeval-021", "ifeval-000", "ifeval-001",
]
RANDOM_LENGTHS = [13, 14, 15, 16, 17, 19, 23, 29, 37, 48, 55, 66, 75, 88, 97, 127]


def adb(*args, timeout=300):
    result = subprocess.run(
        [ADB, "-s", SERIAL, *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"adb exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def shell(command, timeout=300):
    return adb("shell", command, timeout=timeout).stdout


def phone_temperature():
    for line in shell("dumpsys battery").splitlines():
        if line.strip().startswith("temperature:"):
            return int(line.split(":", 1)[1].strip()) / 10.0
    return None


def assert_phone_idle(binary):
    process_names = {
        pathlib.PurePosixPath(binary).name,
        "tensorfold-qualcomm-exl3-forward-4b-policy-dump-v1",
        "tensorfold-qualcomm-exl3-forward-4b-persistent-http-v1",
    }
    processes = shell("pidof " + " ".join(sorted(process_names)) + " 2>/dev/null || true").strip()
    if processes:
        raise RuntimeError(f"Phone B is not idle; TensorFold PIDs: {processes}")


def assert_protected_hashes():
    output = shell("sha256sum " + " ".join(PROTECTED))
    actual = {line.split()[1]: line.split()[0] for line in output.splitlines() if line.strip()}
    for path, expected in PROTECTED.items():
        if actual.get(path) != expected:
            raise RuntimeError(f"protected binary changed: {path}: {actual.get(path)}")


def make_corpus(ifeval_path, seed):
    source = json.loads(pathlib.Path(ifeval_path).read_text(encoding="utf-8"))
    corpus = {sample_id: source[sample_id] for sample_id in SELECTED_IFEVAL}
    rng = random.Random(seed)
    prefix = [248045, 846, 198]
    suffix = [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
    for length in RANDOM_LENGTHS:
        middle = [rng.randrange(0, 151643) for _ in range(length - len(prefix) - len(suffix))]
        corpus[f"random-{length:03d}"] = prefix + middle + suffix
    return corpus


def run_engine(speculative, binary, quiet=True):
    exports = [
        f"LD_LIBRARY_PATH={BUNDLE}/lib", f"ADSP_LIBRARY_PATH={BUNDLE}/lib",
        "GGML_HEXAGON_OPPOLL=1", "GGML_HEXAGON_HMX_PIPELINE_MIN_ROWS=4",
        "TENSORFOLD_Q8_OUTPUT_LAYER_END=24", "TENSORFOLD_FUSED_LINEAR_LAYER_MASK=0",
    ]
    unsets = [
        "TENSORFOLD_MTP_POLICY_DUMP", "TENSORFOLD_MTP_SECOND_MARGIN",
        "TENSORFOLD_CORPUS_LENGTH", "TENSORFOLD_STREAM_TOKENS",
        "TENSORFOLD_PERSISTENT_WORKER",
    ]
    if quiet:
        exports.append("TENSORFOLD_BENCH_QUIET=1")
    else:
        unsets.append("TENSORFOLD_BENCH_QUIET")
    if speculative:
        exports += [
            f"TENSORFOLD_MTP_SIDECAR={MTP}", "TENSORFOLD_MTP_DEPTH=2",
            "TENSORFOLD_MTP_INITIAL_RANK=1", "TENSORFOLD_MTP_FIRST_MARGIN=0.1",
            "TENSORFOLD_TARGET_AMBIGUITY_MARGIN=0.125",
        ]
    else:
        unsets += [
            "TENSORFOLD_MTP_SIDECAR", "TENSORFOLD_MTP_DEPTH",
            "TENSORFOLD_MTP_INITIAL_RANK", "TENSORFOLD_MTP_FIRST_MARGIN",
            "TENSORFOLD_TARGET_AMBIGUITY_MARGIN",
        ]
    command = (
        f"cd {BUNDLE}; export {' '.join(exports)}; unset {' '.join(unsets)}; "
        f"exec {binary} htp {MODEL} 0 6 {CACHE} {MODEL_SHA} 8 0 {MODE} 1 {REMOTE_CORPUS} 2>&1"
    )
    return shell(command, timeout=600)


def json_events(output, kind):
    events = []
    for line in output.splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if event.get("tensorfold") == kind:
            events.append(event)
    return events


def fnv1a_tokens(tokens):
    value = 14695981039346656037
    for token in tokens:
        value ^= token
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:x}"


def serial_hashes(output):
    result = {}
    for event in json_events(output, "qualcomm-exl3-generate"):
        count = event["generated_tokens"]
        result[event["sample_id"]] = fnv1a_tokens(event["tokens"][-count:])
    return result


def speculative_hashes(output, sample_ids):
    events = json_events(output, "qualcomm-exl3-speculative-summary")
    if len(events) != len(sample_ids):
        raise RuntimeError(f"expected {len(sample_ids)} speculative summaries, got {len(events)}")
    return {sample_id: event["token_hash"] for sample_id, event in zip(sample_ids, events)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifeval-prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--binary", default=BINARY)
    parser.add_argument("--seed", type=int, default=0x54464F4C44)
    parser.add_argument("--sample", action="append")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    assert_phone_idle(args.binary)
    assert_protected_hashes()
    temperature_before = phone_temperature()
    corpus = make_corpus(args.ifeval_prompts, args.seed)
    if args.sample:
        missing = set(args.sample) - set(corpus)
        if missing:
            raise ValueError(f"unknown samples: {sorted(missing)}")
        corpus = {sample_id: corpus[sample_id] for sample_id in args.sample}
    with tempfile.TemporaryDirectory() as directory:
        local_corpus = pathlib.Path(directory) / "corpus.json"
        local_corpus.write_text(json.dumps(corpus, separators=(",", ":")), encoding="utf-8")
        adb("push", str(local_corpus), REMOTE_CORPUS, timeout=60)
        try:
            serial_output = run_engine(False, args.binary, not args.verbose)
            speculative_output = run_engine(True, args.binary, not args.verbose)
            serial = serial_hashes(serial_output)
            speculative = speculative_hashes(speculative_output, list(corpus))
        finally:
            shell(f"rm -f {REMOTE_CORPUS}")

    samples = []
    for sample_id, tokens in corpus.items():
        samples.append({
            "sample_id": sample_id,
            "prompt_tokens": len(tokens),
            "serial_token_hash": serial.get(sample_id),
            "speculative_token_hash": speculative.get(sample_id),
            "exact": serial.get(sample_id) == speculative.get(sample_id),
        })
    temperature_after = phone_temperature()
    assert_phone_idle(args.binary)
    report = {
        "device_serial": SERIAL,
        "model_sha256": MODEL_SHA,
        "binary": args.binary,
        "policy": {"depth": 2, "initial_rank": 1, "first_margin": 0.1},
        "random_seed": args.seed,
        "temperature_c_before": temperature_before,
        "temperature_c_after": temperature_after,
        "selected_ifeval_samples": sum(sample_id.startswith("ifeval-") for sample_id in corpus),
        "randomized_samples": sum(sample_id.startswith("random-") for sample_id in corpus),
        "passed": all(sample["exact"] for sample in samples),
        "samples": samples,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".serial.log").write_text(serial_output, encoding="utf-8")
    output.with_suffix(".speculative.log").write_text(speculative_output, encoding="utf-8")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"], "samples": len(samples),
        "temperature_c_before": temperature_before,
        "temperature_c_after": temperature_after,
        "output": str(output),
    }))
    if not report["passed"]:
        for sample in samples:
            if not sample["exact"]:
                print(json.dumps(sample))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
