"""Independent parity verifier: compare safetensors shard and emitted .tfold.

Rebuilds every tensor's sha256 from BOTH files and asserts byte-equivalence.
Writes a pass/fail JSON record. No TFOLD runtime required.
"""
import argparse, hashlib, json, sys
from pathlib import Path

def read_safetensors_header(path):
    with path.open("rb") as f:
        hdr_len = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(hdr_len).decode())
    if "__metadata__" in hdr: hdr.pop("__metadata__")
    return hdr, 8 + hdr_len

def read_tfold_header(path):
    with path.open("rb") as f:
        magic = f.read(5)
        assert magic == b"TFOLD", f"bad magic {magic!r}"
        ver = f.read(1)[0]
        assert ver == 0x02, f"bad version {ver}"
        f.read(6)  # reserved
        hdr_len = int.from_bytes(f.read(4), "little")
        hdr = json.loads(f.read(hdr_len).decode())
    return hdr, 16 + hdr_len

def tensor_sha_from_safetensors(path, name, meta, data_offset):
    s, e = meta["data_offsets"]
    h = hashlib.sha256()
    left = e - s
    with path.open("rb") as f:
        f.seek(data_offset + s)
        while left > 0:
            chunk = f.read(min(left, 1<<20))
            h.update(chunk); left -= len(chunk)
    return h.hexdigest()

def tensor_sha_from_tfold(path, entry):
    h = hashlib.sha256()
    left = entry["nbytes"]
    with path.open("rb") as f:
        f.seek(entry["offset"])
        while left > 0:
            chunk = f.read(min(left, 1<<20))
            h.update(chunk); left -= len(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--tfold", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    st_path = Path(args.safetensors); tf_path = Path(args.tfold)
    st_hdr, st_data_off = read_safetensors_header(st_path)
    tf_hdr, _ = read_tfold_header(tf_path)
    st_tensors = st_hdr
    tf_tensors = {t["name"]: t for t in tf_hdr["tensors"]}

    # Structural checks
    missing = set(st_tensors) - set(tf_tensors)
    extra = set(tf_tensors) - set(st_tensors)
    if missing or extra:
        rep = {"ok": False, "reason": "name_set_mismatch",
               "missing_from_tfold": sorted(missing)[:20],
               "extra_in_tfold": sorted(extra)[:20]}
        Path(args.report).write_text(json.dumps(rep, indent=2))
        print("FAIL: name set mismatch", file=sys.stderr); sys.exit(1)

    # Per-tensor shape + nbytes + sha
    mismatches = []
    total = len(st_tensors)
    for i, (name, meta) in enumerate(st_tensors.items(), start=1):
        if i % 100 == 0: print(f"  {i}/{total}", file=sys.stderr)
        te = tf_tensors[name]
        exp_nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
        checks = []
        if list(meta["shape"]) != list(te["shape"]):
            checks.append(("shape", meta["shape"], te["shape"]))
        if exp_nbytes != te["nbytes"]:
            checks.append(("nbytes", exp_nbytes, te["nbytes"]))
        st_sha = tensor_sha_from_safetensors(st_path, name, meta, st_data_off)
        tf_sha = tensor_sha_from_tfold(tf_path, te)
        if st_sha != tf_sha:
            checks.append(("sha256", st_sha, tf_sha))
        if te.get("sha256") and te["sha256"] != tf_sha:
            checks.append(("manifest_sha256", te["sha256"], tf_sha))
        if checks:
            mismatches.append({"name": name, "checks": checks})

    rep = {
        "ok": len(mismatches) == 0,
        "n_tensors_checked": total,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
    }
    Path(args.report).write_text(json.dumps(rep, indent=2))
    if rep["ok"]:
        print(f"PASS: {total} tensors byte-identical", file=sys.stderr)
    else:
        print(f"FAIL: {len(mismatches)} mismatches (first 20 in report)", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
