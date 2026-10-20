#!/usr/bin/env python3
"""Capture teacher-forced Qualcomm MTP draft/verify events for offline policy simulation."""
import argparse, pathlib, subprocess

ADB = r"D:/projects/phoneproject/.tools/android-sdk/platform-tools/adb.exe"
SERIAL = "e1db9f1d"
BUNDLE = "/data/local/tmp/exl3-full-power"
BINARY = BUNDLE + "/tensorfold-qualcomm-exl3-forward-4b-policy-dump-v1"
MODEL = "/data/local/tmp/tensorfold-models/Qwen3.5-4B-exl3-4.00bpw/model.safetensors"
CACHE = "/data/local/tmp/tensorfold-exl3-cache-4b-v1"
MTP = "/data/local/tmp/tensorfold-mtp-poc/mtp.safetensors"
SHA = "117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1"
MODE = "serve-q4-full-out-q8-q0-head-adaptive-linear"
FIXTURES = {
    "code": [7734,264,2716,12654,709,421,55288,279,76938,8240,321,10033,424,13],
    "chat": [248045,846,198,814,20139,1204,5979,44896,5533,264,21966,303,13983,6163,11,1179,2873,264,2526,33625,3010,13,248046,198,248045,74455,198,248068,271,248069,271],
}


def command(tokens, depth):
    first, suffix = tokens[0], ",".join(map(str, tokens[1:]))
    env = (f"cd {BUNDLE}; export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib "
           f"GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_HMX_PIPELINE_MIN_ROWS=4 "
           f"TENSORFOLD_MTP_SIDECAR={MTP} TENSORFOLD_MTP_DEPTH={depth} "
           f"TENSORFOLD_MTP_INITIAL_RANK=1 TENSORFOLD_MTP_POLICY_DUMP=1 "
           f"TENSORFOLD_Q8_OUTPUT_LAYER_END=24 TENSORFOLD_FUSED_LINEAR_LAYER_MASK=0; "
           "unset TENSORFOLD_BENCH_QUIET TENSORFOLD_STREAM_TOKENS TENSORFOLD_PERSISTENT_WORKER "
           "TENSORFOLD_MTP_FIRST_MARGIN TENSORFOLD_MTP_SECOND_MARGIN; ")
    argv = f"{BINARY} htp {MODEL} {first} 6 {CACHE} {SHA} 256 {suffix} {MODE} 1"
    return env + "exec " + argv + " 2>&1"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True)
    args = ap.parse_args(); out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for name, tokens in FIXTURES.items():
        for depth in (1, 2):
            path = out / f"{name}-d{depth}.jsonl"
            print(path, flush=True)
            with path.open("w", encoding="utf-8") as handle:
                run = subprocess.run([ADB, "-s", SERIAL, "shell", command(tokens, depth)],
                                     stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=240)
            if run.returncode != 0: raise SystemExit(f"{name} depth {depth}: adb rc {run.returncode}")


if __name__ == "__main__": main()
