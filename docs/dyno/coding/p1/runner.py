"""Stage P1 - coding track pilot.

Per attempt:
  1. Make isolated git clone + checkout base_commit.
  2. Apply test_patch so failing test is present.
  3. Ask model for a unified diff.
  4. git apply --index, run pytest; record oracle verdict.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_MODEL_DIR", "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw")

PATCH_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
UNIFIED_DIFF_RE = re.compile(r"(diff --git .*)", re.DOTALL)
MINIMAL_DIFF_RE = re.compile(r"(--- a/\S+\s*\n\+\+\+ b/\S+.*)", re.DOTALL)

def extract_patch(raw_text):
    """Accept either `diff --git` or bare `--- a/... +++ b/...` diffs.
    If we only have the minimal form, synthesize a `diff --git a/X b/X` prelude so
    `git apply --index` will accept it."""
    body = None
    m = PATCH_FENCE_RE.search(raw_text)
    if m:
        body = m.group(1).rstrip() + "\n"
    if body is None:
        m = UNIFIED_DIFF_RE.search(raw_text)
        if m:
            body = m.group(1).rstrip() + "\n"
    if body is None:
        m = MINIMAL_DIFF_RE.search(raw_text)
        if m:
            body = m.group(1).rstrip() + "\n"
    if body is None:
        return None
    # If we have a minimal diff, synthesize the `diff --git` header line
    if not body.lstrip().startswith("diff --git"):
        mm = re.match(r"\s*--- a/(\S+)\s*\n\+\+\+ b/(\S+)", body)
        if not mm:
            return None
        body = f"diff --git a/{mm.group(1)} b/{mm.group(2)}\n" + body.lstrip()
    return body

def sh(cmd, cwd=None, timeout=None, env=None):
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8","replace")
    except subprocess.TimeoutExpired as e:
        return -1, f"TIMEOUT after {timeout}s: {e}"

def make_prompt(instance, failing_file_content):
    return f"""You are editing the repository {instance['repo']} at commit {instance['base_commit']}.

TASK:
{instance['problem_statement'].strip()}

FAILING TEST (already in the test file):
{instance['failing_test_body']}

RELEVANT FILE ({instance['target_file']}):
```python
{failing_file_content}
```

Produce a MINIMAL unified diff (git-format) that makes the failing test pass without breaking other tests. Output ONLY the diff inside a ```diff ...``` code fence. Do not explain.
"""

def run_one(instance, workspace_root, flask_base, venv, mode, seed, sampler_name, gen_fn, fh):
    attempt_dir = workspace_root / f"attempt_{mode}_{seed}"
    if attempt_dir.exists():
        sh(f"chmod -R u+w {attempt_dir} 2>/dev/null; rm -rf {attempt_dir}")

    rc, out = sh(f"git clone --quiet {flask_base} {attempt_dir}")
    if rc != 0:
        rec = {"mode": mode, "seed": seed, "status": "clone_failed", "out_head": out[:400]}
        fh.write(json.dumps(rec)+"\n"); fh.flush(); return
    rc, out = sh(f"git checkout --quiet {instance['base_commit']}", cwd=attempt_dir)

    with (attempt_dir/"_test_patch.diff").open("w") as g: g.write(instance["test_patch"])
    rc, out = sh("git apply --index _test_patch.diff", cwd=attempt_dir)
    if rc != 0:
        rec = {"mode": mode, "seed": seed, "status": "test_patch_apply_failed", "out_head": out[:400]}
        fh.write(json.dumps(rec)+"\n"); fh.flush(); return

    with (attempt_dir/instance["target_file"]).open() as g:
        file_content = g.read()
    prompt = make_prompt(instance, file_content)

    t0 = time.time()
    out_text, ntok, ttft = gen_fn(prompt, sampler_name, seed)
    gen_elapsed = time.time() - t0

    patch = extract_patch(out_text)
    rec = {
        "mode": mode, "seed": seed, "sampler": sampler_name,
        "gen_elapsed_s": round(gen_elapsed,3), "ttft_s": ttft, "n_tokens": ntok,
        "prompt_chars": len(prompt),
        "raw_output_head": out_text[:800],
        "patch_extracted": patch is not None,
    }
    if patch is None:
        rec["status"] = "malformed_patch"
        fh.write(json.dumps(rec)+"\n"); fh.flush(); return

    with (attempt_dir/"_model_patch.diff").open("w") as g: g.write(patch)
    rc, out = sh("git apply --index _model_patch.diff", cwd=attempt_dir)
    rec["patch_apply_rc"] = rc
    rec["patch_apply_out_head"] = out[:400]
    rec["patch_lines"] = patch.count("\n")
    if rc != 0:
        rec["status"] = "patch_apply_failed"
        fh.write(json.dumps(rec)+"\n"); fh.flush(); return

    env = os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"] = "1"

    f2p_list = instance["FAIL_TO_PASS"]
    f2p_target = " ".join(f'"{t}"' for t in f2p_list)
    t1 = time.time()
    rc_f2p, out_f2p = sh(
        f'. {venv}/bin/activate && pytest -q --no-header --tb=no -W "ignore::DeprecationWarning" {f2p_target}',
        cwd=attempt_dir, timeout=180, env=env)
    rec["f2p_elapsed_s"] = round(time.time()-t1, 3)
    rec["f2p_rc"] = rc_f2p
    rec["f2p_out_tail"] = out_f2p[-500:]
    f2p_passed = rc_f2p == 0

    p2p_list = instance["PASS_TO_PASS"]
    p2p_target = " ".join(f'"{t}"' for t in p2p_list[:60])
    t2 = time.time()
    rc_p2p, out_p2p = sh(
        f'. {venv}/bin/activate && pytest -q --no-header --tb=no -W "ignore::DeprecationWarning" {p2p_target}',
        cwd=attempt_dir, timeout=300, env=env)
    rec["p2p_elapsed_s"] = round(time.time()-t2, 3)
    rec["p2p_rc"] = rc_p2p
    rec["p2p_out_tail"] = out_p2p[-500:]
    p2p_passed = rc_p2p == 0

    rec["resolved"] = f2p_passed and p2p_passed
    rec["status"] = "resolved" if rec["resolved"] else (
        "f2p_fail_p2p_fail" if (not f2p_passed and not p2p_passed)
        else "f2p_fail" if not f2p_passed
        else "p2p_fail_regression"
    )
    fh.write(json.dumps(rec)+"\n"); fh.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--flask-base", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    args = ap.parse_args()

    instance = json.loads(Path(args.instance).read_text())
    workspace = Path(args.workspace); workspace.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
    from exllamav3.generator.sampler import DefaultSampler, GreedySampler
    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
    cache = Cache(model, max_num_tokens=8192); model.load()
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    IM_END = tok.single_id("<|im_end|>"); ENDTXT = tok.single_id("<|endoftext|>")
    eos = [i for i in [IM_END, ENDTXT] if i is not None]

    def gen_fn(prompt, sampler_name, seed):
        sampler = GreedySampler() if sampler_name == "greedy" else DefaultSampler()
        ids = tok.hf_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)
        t0 = time.time(); t_first = None
        job = Job(input_ids=ids, max_new_tokens=1500, sampler=sampler, seed=seed, stop_conditions=eos)
        gen.enqueue(job)
        text=""; ntok=0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming":
                    if t_first is None and r.get("text"): t_first = time.time()
                    text += r.get("text",""); ntok += 1
                    if time.time() - t0 > 180:
                        ttft = round(t_first-t0,3) if t_first else None
                        return text, ntok, ttft
        ttft = round(t_first-t0, 3) if t_first else None
        return text, ntok, ttft

    seeds_sampled = [8001, 8002, 8003]
    with out_path.open("a") as fh:
        run_one(instance, workspace, args.flask_base, args.venv, "greedy", 8000, "greedy", gen_fn, fh)
        for s in seeds_sampled:
            run_one(instance, workspace, args.flask_base, args.venv, "sampled", s, "default", gen_fn, fh)

    print(f"done. receipts at {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
