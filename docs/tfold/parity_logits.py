"""Run tf-qwen35-cuda twice (safetensors dir, then .tfold dir) on the same prompts
and compare final logits + argmax token sequences bit-exactly and numerically.

The stage subcommand writes a binary dump of logit rows. We ask both runs to write
to distinct output paths, then open each as the concatenated bf16 logit array,
align by (window, row) and compute max/mean abs + relative deltas + argmax diffs.
"""
import argparse, json, os, struct, subprocess, sys, time
from pathlib import Path

def run_stage(binary: str, model_dir: str, out_bin: str, tokens: int, env: dict):
    environ = os.environ.copy()
    environ.update(env)
    environ["TF_PIPE_TOKENS"] = str(tokens)
    t0 = time.time()
    r = subprocess.run([binary, model_dir, "stage", out_bin],
                       env=environ, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    return {"rc": r.returncode, "dt_s": dt, "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:]}

def read_bf16_rows(path: Path, vocab: int):
    """Parse the stage dump - it's little-endian bf16 pairs, V entries per row."""
    data = path.read_bytes()
    nbytes = len(data)
    bytes_per_row = vocab * 2
    assert nbytes % bytes_per_row == 0, f"{nbytes} not divisible by vocab*2={bytes_per_row}"
    rows = nbytes // bytes_per_row
    import array
    # Parse as u16 then convert to f32 via bf16-to-f32 (upper 16 bits = u16, lower zero)
    arr = array.array('H')
    arr.frombytes(data)
    return arr, rows, vocab

def bf16_to_f32(u16_val: int) -> float:
    return struct.unpack('<f', struct.pack('<I', u16_val << 16))[0]

def compare_logits(a_arr, b_arr, rows, vocab):
    max_abs = 0.0; sum_abs = 0.0; count = 0
    max_rel = 0.0
    bit_identical = 0; argmax_mismatches = 0
    for row in range(rows):
        base = row * vocab
        amax_a = 0; amax_b = 0
        best_a = -float("inf"); best_b = -float("inf")
        for i in range(vocab):
            ua = a_arr[base + i]; ub = b_arr[base + i]
            if ua == ub: bit_identical += 1
            fa = bf16_to_f32(ua); fb = bf16_to_f32(ub)
            d = abs(fa - fb)
            if d > max_abs: max_abs = d
            sum_abs += d; count += 1
            denom = max(abs(fa), 1e-6)
            r = d / denom
            if r > max_rel: max_rel = r
            if fa > best_a: best_a = fa; amax_a = i
            if fb > best_b: best_b = fb; amax_b = i
        if amax_a != amax_b: argmax_mismatches += 1
    return {
        "rows": rows,
        "vocab": vocab,
        "max_abs_delta": max_abs,
        "mean_abs_delta": sum_abs / max(count, 1),
        "max_rel_delta": max_rel,
        "bit_identical_pct": 100.0 * bit_identical / max(count, 1),
        "argmax_mismatches": argmax_mismatches,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--safetensors-dir", required=True)
    ap.add_argument("--tfold-dir", required=True)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=5)
    ap.add_argument("--out", required=True, help="json report path")
    args = ap.parse_args()

    sft_out = Path("/tmp/parity_stage_sft.bin")
    tf_out  = Path("/tmp/parity_stage_tf.bin")

    print("[1/4] running safetensors path", file=sys.stderr)
    r_sft = run_stage(args.binary, args.safetensors_dir, str(sft_out), args.tokens, {"TF_NORM_AUDIT": "1"})
    print("[2/4] running tfold path", file=sys.stderr)
    r_tf  = run_stage(args.binary, args.tfold_dir,       str(tf_out),  args.tokens, {"TF_NORM_AUDIT": "1"})

    rep = {"safetensors": r_sft, "tfold": r_tf}
    if r_sft["rc"] != 0 or r_tf["rc"] != 0 or not sft_out.exists() or not tf_out.exists():
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("abort: one or both runs failed", file=sys.stderr); sys.exit(1)

    print("[3/4] reading logit dumps", file=sys.stderr)
    a_arr, rows_a, V = read_bf16_rows(sft_out, args.vocab)
    b_arr, rows_b, _ = read_bf16_rows(tf_out,  args.vocab)
    rep["rows_safetensors"] = rows_a
    rep["rows_tfold"] = rows_b
    assert rows_a == rows_b, f"row count drift: {rows_a} vs {rows_b}"

    print("[4/4] comparing", file=sys.stderr)
    rep["comparison"] = compare_logits(a_arr, b_arr, rows_a, V)
    Path(args.out).write_text(json.dumps(rep, indent=2, default=str))
    print("report:", args.out, file=sys.stderr)
    print("  bit-identical bf16 slots:", f"{rep['comparison']['bit_identical_pct']:.4f}%")
    print("  max abs delta:", rep["comparison"]["max_abs_delta"])
    print("  argmax mismatches:", rep["comparison"]["argmax_mismatches"])

if __name__ == "__main__":
    main()
