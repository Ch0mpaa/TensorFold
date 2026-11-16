"""Stage P1b - coding track, two lanes compared on the same instance.

Lane RAW:      model emits a unified diff directly (reuses P1 harness logic).
Lane ASSISTED: model emits a JSON edit-spec; harness applies the spec to the
               file literally, then generates the patch with `git diff` and
               validates it with `git apply --check` in a CLEAN worktree.

Scored INDEPENDENTLY. We do NOT retroactively count repaired patches as raw
successes. Each attempt records lane, raw_output, {patch_valid, resolved, ...}.

Edit-spec protocol (ONE JSON object per attempt):
  {
    "file": "src/flask/blueprints.py",
    "action": "insert_before" | "insert_after" | "replace",
    "anchor": "<verbatim line already in the file; MUST match exactly one line>",
    "payload": "<new code to insert (for insert_*) or replacement (for replace)>"
  }
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_MODEL_DIR", "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw")

PATCH_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
UNIFIED_DIFF_RE = re.compile(r"(diff --git .*)", re.DOTALL)
MINIMAL_DIFF_RE = re.compile(r"(--- a/\S+\s*\n\+\+\+ b/\S+.*)", re.DOTALL)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*```", re.DOTALL)
JSON_OBJECT_RE = re.compile(r"(\{[^{}]*\"file\"[^{}]*\"payload\"[^{}]*\})", re.DOTALL)

def extract_patch(raw_text):
    body = None
    m = PATCH_FENCE_RE.search(raw_text)
    if m: body = m.group(1).rstrip() + "\n"
    if body is None:
        m = UNIFIED_DIFF_RE.search(raw_text)
        if m: body = m.group(1).rstrip() + "\n"
    if body is None:
        m = MINIMAL_DIFF_RE.search(raw_text)
        if m: body = m.group(1).rstrip() + "\n"
    if body is None: return None
    if not body.lstrip().startswith("diff --git"):
        mm = re.match(r"\s*--- a/(\S+)\s*\n\+\+\+ b/(\S+)", body)
        if not mm: return None
        body = f"diff --git a/{mm.group(1)} b/{mm.group(2)}\n" + body.lstrip()
    return body

def extract_edit_spec(raw_text):
    m = JSON_FENCE_RE.search(raw_text)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    # fallback: first {...} with 'file' and 'payload'
    m = JSON_OBJECT_RE.search(raw_text)
    if m:
        try: return json.loads(m.group(1))
        except Exception: return None
    # looser: scan for balanced braces starting at first '{'
    idx = raw_text.find("{")
    while idx >= 0:
        depth = 0
        for j in range(idx, len(raw_text)):
            c = raw_text[j]
            if c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw_text[idx:j+1])
                        if isinstance(obj, dict) and "file" in obj and "payload" in obj:
                            return obj
                    except Exception: pass
                    break
        idx = raw_text.find("{", idx+1)
    return None

def sh(cmd, cwd=None, timeout=None, env=None):
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8","replace")
    except subprocess.TimeoutExpired as e:
        return -1, f"TIMEOUT {timeout}s"

def make_raw_prompt(instance, file_content):
    return f"""You are editing the repository {instance['repo']} at commit {instance['base_commit']}.

TASK:
{instance['problem_statement'].strip()}

FAILING TEST (already in the test file):
{instance['failing_test_body']}

RELEVANT FILE ({instance['target_file']}):
```python
{file_content}
```

Produce a MINIMAL unified diff (git-format) that makes the failing test pass without breaking other tests. Output ONLY the diff inside a ```diff ...``` code fence. Do not explain.
"""

def make_assisted_prompt(instance, file_content):
    return f"""You are editing the repository {instance['repo']} at commit {instance['base_commit']}.

TASK:
{instance['problem_statement'].strip()}

FAILING TEST (already in the test file):
{instance['failing_test_body']}

RELEVANT FILE ({instance['target_file']}):
```python
{file_content}
```

Respond with ONE JSON object describing the edit. Schema:
```json
{{
  "file":    "{instance['target_file']}",
  "action":  "insert_before" | "insert_after" | "replace",
  "anchor":  "<copy exactly ONE line that appears verbatim in the file above; the edit happens relative to that line>",
  "payload": "<the new code to insert or substitute; use \\n for newlines>"
}}
```

The anchor may span MULTIPLE lines (just paste the lines verbatim). It MUST appear EXACTLY ONCE in the file.

Output ONLY the JSON object inside a ```json ...``` fence. Do not explain.
"""

def apply_edit_spec(spec, attempt_dir, instance):
    """Apply the JSON spec literally; return (ok, reason).
    Anchor is located by verbatim substring match in the target file.
    Must appear exactly once."""
    if not isinstance(spec, dict):
        return False, "spec_not_object"
    for k in ("file", "action", "anchor", "payload"):
        if k not in spec:
            return False, f"missing_{k}"
    if spec["action"] not in ("insert_before","insert_after","replace"):
        return False, f"bad_action_{spec['action']}"
    target = attempt_dir / spec["file"]
    if not target.exists():
        return False, f"file_not_found_{spec['file']}"
    file_text = target.read_text()
    anchor = spec["anchor"]
    if not anchor:
        return False, "empty_anchor"
    count = file_text.count(anchor)
    if count == 0:
        return False, "anchor_not_found"
    if count > 1:
        return False, f"anchor_ambiguous_{count}_matches"
    pos = file_text.find(anchor)
    payload = spec["payload"]
    # Add a trailing newline if missing (keeps line-oriented structure)
    if payload and not payload.endswith("\n"):
        payload = payload + "\n"
    if spec["action"] == "insert_before":
        new_text = file_text[:pos] + payload + file_text[pos:]
    elif spec["action"] == "insert_after":
        new_text = file_text[:pos + len(anchor)] + payload + file_text[pos + len(anchor):]
    else:  # replace
        new_text = file_text[:pos] + payload + file_text[pos + len(anchor):]
    target.write_text(new_text)
    return True, "ok"

def run_one(instance, workspace_root, flask_base, venv, lane, seed, sampler_name, gen_fn, fh):
    attempt_dir = workspace_root / f"attempt_{lane}_{sampler_name}_{seed}"
    if attempt_dir.exists():
        sh(f"chmod -R u+w {attempt_dir} 2>/dev/null; rm -rf {attempt_dir}")

    sh(f"git clone --quiet {flask_base} {attempt_dir}")
    sh(f"git checkout --quiet {instance['base_commit']}", cwd=attempt_dir)
    with (attempt_dir/"_test_patch.diff").open("w") as g: g.write(instance["test_patch"])
    sh("git apply --index _test_patch.diff", cwd=attempt_dir)
    # Commit the test_patch so HEAD is clean before the model edits.
    # git diff HEAD will then capture ONLY the model's change.
    sh("git -c user.email=dyno@local -c user.name=dyno commit -q -m 'stage: apply test_patch'", cwd=attempt_dir)

    file_content = (attempt_dir/instance["target_file"]).read_text()
    if lane == "raw":
        prompt = make_raw_prompt(instance, file_content)
    else:
        prompt = make_assisted_prompt(instance, file_content)

    t0 = time.time()
    out_text, ntok, ttft = gen_fn(prompt, sampler_name, seed)
    gen_elapsed = time.time() - t0

    rec = {
        "lane": lane, "sampler": sampler_name, "seed": seed,
        "gen_elapsed_s": round(gen_elapsed,3), "ttft_s": ttft, "n_tokens": ntok,
        "prompt_chars": len(prompt),
        "raw_output_head": out_text[:800],
    }

    if lane == "raw":
        patch = extract_patch(out_text)
        rec["patch_extracted"] = patch is not None
        if patch is None:
            rec["status"] = "malformed_patch"
            fh.write(json.dumps(rec)+"\n"); fh.flush(); return
        with (attempt_dir/"_model_patch.diff").open("w") as g: g.write(patch)
        rc, out = sh("git apply --check _model_patch.diff", cwd=attempt_dir)
        rec["patch_valid_check"] = rc == 0
        rec["check_out_head"] = out[:300]
        if rc != 0:
            rec["status"] = "patch_invalid_check"
            fh.write(json.dumps(rec)+"\n"); fh.flush(); return
        rc, _ = sh("git apply --index _model_patch.diff", cwd=attempt_dir)
        rec["patch_apply_rc"] = rc
        if rc != 0:
            rec["status"] = "patch_apply_failed"
            fh.write(json.dumps(rec)+"\n"); fh.flush(); return

    else:  # assisted
        spec = extract_edit_spec(out_text)
        rec["spec_extracted"] = spec is not None
        if spec is None:
            rec["status"] = "malformed_spec"
            fh.write(json.dumps(rec)+"\n"); fh.flush(); return
        rec["spec_head"] = {k: (v[:200] if isinstance(v,str) else v) for k,v in spec.items()}
        ok, reason = apply_edit_spec(spec, attempt_dir, instance)
        rec["spec_apply_ok"] = ok
        rec["spec_apply_reason"] = reason
        if not ok:
            rec["status"] = f"spec_apply_failed:{reason}"
            fh.write(json.dumps(rec)+"\n"); fh.flush(); return
        # Generate a patch from the resulting diff and validate separately
        rc, patch_body = sh("git diff --no-color HEAD", cwd=attempt_dir)
        rec["generated_patch_bytes"] = len(patch_body)
        with (attempt_dir/"_generated_patch.diff").open("w") as g: g.write(patch_body)
        # Reset the file and re-apply via git apply --check on a clean worktree
        sh(f"git checkout {instance['target_file']}", cwd=attempt_dir)
        rc_chk, out_chk = sh("git apply --check _generated_patch.diff", cwd=attempt_dir)
        rec["patch_valid_check"] = rc_chk == 0
        rec["check_out_head"] = out_chk[:300]
        if rc_chk != 0:
            rec["status"] = "generated_patch_invalid"
            fh.write(json.dumps(rec)+"\n"); fh.flush(); return
        rc_app, _ = sh("git apply --index _generated_patch.diff", cwd=attempt_dir)
        rec["patch_apply_rc"] = rc_app
        if rc_app != 0:
            rec["status"] = "generated_patch_apply_failed"
            fh.write(json.dumps(rec)+"\n"); fh.flush(); return

    # oracle
    env = os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"] = "1"
    f2p_list = instance["FAIL_TO_PASS"]
    f2p_target = " ".join(f'"{t}"' for t in f2p_list)
    t1 = time.time()
    rc_f2p, out_f2p = sh(
        f'. {venv}/bin/activate && pytest -q --no-header --tb=no -W "ignore::DeprecationWarning" {f2p_target}',
        cwd=attempt_dir, timeout=180, env=env)
    rec["f2p_elapsed_s"] = round(time.time()-t1, 3)
    rec["f2p_rc"] = rc_f2p
    rec["f2p_out_tail"] = out_f2p[-400:]

    p2p_list = instance["PASS_TO_PASS"]
    p2p_target = " ".join(f'"{t}"' for t in p2p_list[:60])
    t2 = time.time()
    rc_p2p, out_p2p = sh(
        f'. {venv}/bin/activate && pytest -q --no-header --tb=no -W "ignore::DeprecationWarning" {p2p_target}',
        cwd=attempt_dir, timeout=300, env=env)
    rec["p2p_elapsed_s"] = round(time.time()-t2, 3)
    rec["p2p_rc"] = rc_p2p
    rec["p2p_out_tail"] = out_p2p[-400:]

    f2p_passed = rc_f2p == 0
    p2p_passed = rc_p2p == 0
    rec["resolved"] = f2p_passed and p2p_passed
    rec["status"] = "resolved" if rec["resolved"] else (
        "f2p_fail_p2p_fail" if (not f2p_passed and not p2p_passed)
        else "f2p_fail" if not f2p_passed
        else "p2p_fail_regression")
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

    with out_path.open("a") as fh:
        # Same 4 seeds for both lanes, matched by (lane, sampler, seed)
        for lane in ("raw", "assisted"):
            run_one(instance, workspace, args.flask_base, args.venv, lane, 8000, "greedy", gen_fn, fh)
            for s in (8001, 8002, 8003):
                run_one(instance, workspace, args.flask_base, args.venv, lane, s, "default", gen_fn, fh)

    print(f"done. receipts at {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
