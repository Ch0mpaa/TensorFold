"""P1c intervention: bounded read -> edit -> test -> diagnose -> retry workflow.

Up to 3 total edit attempts per task:
  Turn 1: model emits edit-spec (JSON {file, action, anchor, payload}).
  Harness applies spec in an isolated git worktree, commits, runs FAIL_TO_PASS test.
  Turn 2 (if fail): model sees {pytest_tail, current_file_window_around_edit_site},
    emits a new edit-spec OR {"tool":"finish","args":{}} to stop.
  Turn 3 (if still fail): same again.
  Then oracle: FAIL_TO_PASS + PASS_TO_PASS.

The model MAY emit `finish` early to stop retrying. Retries are CAPPED at 2.

Scored INDEPENDENTLY of raw + assisted baselines from P1b (preserved).
Oracle is pytest only — NO LLM judgment.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_MODEL_DIR", "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw")

# Edit-spec schema used to constrain model output via LLGuidanceFilter
EDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action"],
    "properties": {
        "action": {"type": "string", "enum": ["insert_before", "insert_after", "replace", "finish"]},
        "file":    {"type": "string"},
        "anchor":  {"type": "string"},
        "payload": {"type": "string"},
    },
}

def sh(cmd, cwd=None, timeout=None, env=None):
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8","replace")
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT {timeout}s"

def apply_spec(spec, attempt_dir, target_file):
    f = spec.get("file") or target_file
    path = attempt_dir / f
    if not path.exists():
        return False, f"file_not_found:{f}"
    action = spec.get("action")
    if action not in ("insert_before", "insert_after", "replace"):
        return False, f"bad_action:{action}"
    anchor = spec.get("anchor", "")
    payload = spec.get("payload", "")
    if not anchor:
        return False, "empty_anchor"
    file_text = path.read_text()
    count = file_text.count(anchor)
    if count == 0:
        return False, "anchor_not_found"
    if count > 1:
        return False, f"anchor_ambiguous:{count}"
    pos = file_text.find(anchor)
    if payload and not payload.endswith("\n"):
        payload = payload + "\n"
    if action == "insert_before":
        new_text = file_text[:pos] + payload + file_text[pos:]
    elif action == "insert_after":
        new_text = file_text[:pos + len(anchor)] + payload + file_text[pos + len(anchor):]
    else:  # replace
        new_text = file_text[:pos] + payload + file_text[pos + len(anchor):]
    path.write_text(new_text)
    return True, "ok"

def file_window(path, around_text, max_lines=40):
    """Return a textual window of the file for diagnosis, centered on around_text if present."""
    text = Path(path).read_text()
    lines = text.splitlines()
    center_line = 0
    if around_text:
        for i, ln in enumerate(lines):
            if around_text[:40] in ln:
                center_line = i; break
    lo = max(0, center_line - max_lines // 2)
    hi = min(len(lines), lo + max_lines)
    return "\n".join(f"{i+1:4d} {l}" for i, l in enumerate(lines[lo:hi], start=lo))

def seed_worktree(instance, attempt_dir, flask_base):
    if attempt_dir.exists():
        sh(f"chmod -R u+w {attempt_dir} 2>/dev/null; rm -rf {attempt_dir}")
    sh(f"git clone --quiet {flask_base} {attempt_dir}")
    sh(f"git checkout --quiet {instance['base_commit']}", cwd=attempt_dir)
    with (attempt_dir/"_test_patch.diff").open("w") as g: g.write(instance["test_patch"])
    sh("git apply --index _test_patch.diff", cwd=attempt_dir)
    sh("git -c user.email=dyno@local -c user.name=dyno commit -q -m 'stage: test_patch'", cwd=attempt_dir)

def reset_worktree(attempt_dir, target_file):
    """Revert any uncommitted changes so this turn's edit applies to pristine state."""
    sh("git checkout -- .", cwd=attempt_dir)
    sh("git clean -fdq", cwd=attempt_dir)

def run_tests(attempt_dir, venv, test_targets, test_cmd, timeout=180):
    env = os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = f"{attempt_dir}:" + env.get("PYTHONPATH", "")
    cmd = f'. {venv}/bin/activate && {test_cmd.format(targets=test_targets)}'
    rc, out = sh(cmd, cwd=attempt_dir, timeout=timeout, env=env)
    return rc, out

def make_prompt(instance, file_content, previous_attempts_log=None):
    base = f"""You are editing {instance['repo']} at commit {instance['base_commit']}.

TASK:
{instance['problem_statement'].strip()}

FAILING TEST (already in test file):
{instance['failing_test_body']}

RELEVANT FILE ({instance['target_file']}):
```python
{file_content}
```

Respond with ONE JSON object describing a MINIMAL edit. Schema:
{{
  "action":  "insert_before" | "insert_after" | "replace" | "finish",
  "file":    "{instance['target_file']}",
  "anchor":  "<copy exactly 1-3 lines that appear verbatim in the file; must be unique>",
  "payload": "<the new code to insert or substitute; use \\n for newlines>"
}}

Output ONLY the JSON object. Do not explain.
"""
    if previous_attempts_log:
        base += "\n\nPREVIOUS ATTEMPTS:\n" + previous_attempts_log + \
                "\n\nYour previous attempt(s) did not make the failing test pass. Try a different edit."
    return base

def run_one(instance, workspace_root, flask_base, venv, seed, sampler_name, gen_fn, fh):
    attempt_dir = workspace_root / f"attempt_edittest_{sampler_name}_{seed}"
    seed_worktree(instance, attempt_dir, flask_base)

    target_file = instance["target_file"]
    f2p_targets = " ".join(f'"{t}"' for t in instance["FAIL_TO_PASS"])
    p2p_targets = " ".join(f'"{t}"' for t in instance["PASS_TO_PASS"][:60])
    f2p_cmd = instance.get("test_cmd_f2p", 'pytest -q --no-header --tb=short -W "ignore::DeprecationWarning" {targets}')
    p2p_cmd = instance.get("test_cmd_p2p", 'pytest -q --no-header --tb=short -W "ignore::DeprecationWarning" {targets}')

    attempts_log = ""
    per_attempt = []
    resolved = False
    final_pytest_tail = ""
    total_gen_s = 0.0
    total_tokens = 0
    t_start = time.time()
    model_finished = False

    for turn in range(3):
        file_content = (attempt_dir/target_file).read_text()
        prompt = make_prompt(instance, file_content, attempts_log if turn > 0 else None)
        t0 = time.time()
        out_text, ntok = gen_fn(prompt, sampler_name, seed + turn)
        total_gen_s += time.time() - t0; total_tokens += ntok

        try:
            spec = json.loads(out_text.strip())
        except Exception:
            try:
                i = out_text.find("{"); j = out_text.rfind("}")
                spec = json.loads(out_text[i:j+1])
            except Exception:
                per_attempt.append({"turn": turn, "status": "parse_fail", "out_head": out_text[:200]})
                attempts_log += f"\nTurn {turn}: parse_fail"
                break

        if spec.get("action") == "finish":
            model_finished = True
            per_attempt.append({"turn": turn, "status": "finish_early", "spec_action": "finish"})
            break

        # Reset worktree to pristine so this turn's edit is independent (diagnose from clean state)
        reset_worktree(attempt_dir, target_file)
        ok, reason = apply_spec(spec, attempt_dir, target_file)
        if not ok:
            per_attempt.append({"turn": turn, "status": "spec_apply_failed", "reason": reason,
                                "spec_head": {k: (v[:200] if isinstance(v,str) else v) for k,v in spec.items()}})
            attempts_log += f"\nTurn {turn}: spec apply failed ({reason})"
            continue

        # Generate patch for the oracle (equivalence check)
        rc_diff, patch_body = sh("git diff --no-color HEAD", cwd=attempt_dir)
        # Test fail_to_pass only (quick signal for retry)
        rc_f2p, out_f2p = run_tests(attempt_dir, venv, f2p_targets, f2p_cmd, timeout=60)
        f2p_tail = out_f2p[-400:]
        per_attempt.append({
            "turn": turn, "status": "ran_test",
            "spec_action": spec.get("action"),
            "spec_file": spec.get("file"),
            "spec_anchor_head": (spec.get("anchor","") or "")[:150],
            "spec_payload_head": (spec.get("payload","") or "")[:200],
            "patch_bytes": len(patch_body),
            "f2p_rc": rc_f2p,
            "f2p_tail": f2p_tail,
        })
        final_pytest_tail = f2p_tail
        if rc_f2p == 0:
            # fail_to_pass passed; check pass_to_pass
            rc_p2p, out_p2p = run_tests(attempt_dir, venv, p2p_targets, p2p_cmd, timeout=180)
            per_attempt[-1]["p2p_rc"] = rc_p2p
            per_attempt[-1]["p2p_tail"] = out_p2p[-400:]
            if rc_p2p == 0:
                resolved = True
                break
            else:
                attempts_log += f"\nTurn {turn}: fail_to_pass PASSED but pass_to_pass REGRESSED. tail: {out_p2p[-300:]}"
        else:
            attempts_log += f"\nTurn {turn}: fail_to_pass FAILED. pytest tail:\n{f2p_tail}"

    elapsed = time.time() - t_start
    rec = {
        "instance_id": instance["instance_id"],
        "lane": "edittest",
        "sampler": sampler_name,
        "seed": seed,
        "resolved": resolved,
        "model_finished_early": model_finished,
        "turns_used": len(per_attempt),
        "per_attempt": per_attempt,
        "final_pytest_tail": final_pytest_tail,
        "total_gen_s": round(total_gen_s, 3),
        "total_tokens": total_tokens,
        "total_elapsed_s": round(elapsed, 3),
    }
    fh.write(json.dumps(rec) + "\n"); fh.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--repo-base", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--seed-start", type=int, default=8000)
    ap.add_argument("--n-sampled", type=int, default=3)
    args = ap.parse_args()

    instance = json.loads(Path(args.instance).read_text())
    workspace = Path(args.workspace); workspace.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
    from exllamav3.generator.sampler import DefaultSampler, GreedySampler
    from exllamav3.generator.filter import LLGuidanceFilter
    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
    cache = Cache(model, max_num_tokens=8192); model.load()
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    IM_END = tok.single_id("<|im_end|>"); ENDTXT = tok.single_id("<|endoftext|>")
    eos = [i for i in [IM_END, ENDTXT] if i is not None]

    def gen_fn(prompt, sampler_name, seed):
        sampler = GreedySampler() if sampler_name == "greedy" else DefaultSampler()
        filt = LLGuidanceFilter(tokenizer=tok, json_schema=EDIT_SCHEMA, eos_after_completed=True)
        ids = tok.hf_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)
        t0 = time.time()
        job = Job(input_ids=ids, max_new_tokens=1500, sampler=sampler, seed=seed, stop_conditions=eos, filters=[filt])
        gen.enqueue(job)
        text=""; ntok=0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming":
                    text += r.get("text",""); ntok += 1
                    if time.time() - t0 > 180:
                        return text, ntok
        return text, ntok

    with out_path.open("a") as fh:
        # 1 greedy + N sampled
        run_one(instance, workspace, args.repo_base, args.venv, args.seed_start, "greedy", gen_fn, fh)
        for i in range(args.n_sampled):
            run_one(instance, workspace, args.repo_base, args.venv, args.seed_start + 1 + i, "default", gen_fn, fh)

    print(f"done. receipts at {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
