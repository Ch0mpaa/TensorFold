"""Stage P2b - agentic track, two lanes compared on the same task.

Lane RAW:         model must emit `TOOL: {json}` free-form.
Lane CONSTRAINED: exllamav3 LLGuidanceFilter enforces a JSON schema on the
                  model's turn. The entire assistant turn MUST be a JSON
                  object matching the tool-call schema. The harness no longer
                  needs a regex parser for this lane.

Scored INDEPENDENTLY. We do NOT silently repair malformed baseline calls.
"""
import argparse, json, os, re, shlex, subprocess, sys, time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_MODEL_DIR", "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw")

TOOL_LINE_RE = re.compile(r"^TOOL:\s*(\{.*\})\s*$", re.MULTILINE)

# JSON schema for a single tool call (enforced in constrained lane)
TOOL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tool", "args"],
    "properties": {
        "tool": {"type": "string", "enum": ["shell", "read_file", "write_file", "grep", "finish"]},
        "args": {
            "type": "object",
            "additionalProperties": True
        }
    }
}

def sh(cmd, cwd=None, timeout=15, env=None):
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8","replace")[-2000:]
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT {timeout}s"

def seed_workspace(task, wd):
    if wd.exists():
        sh(f"chmod -R u+w {wd} 2>/dev/null; rm -rf {wd}")
    wd.mkdir(parents=True)
    for rel, content in task["seed_repo_files"].items():
        p = wd / rel; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

def exec_tool(tool, args, wd):
    try:
        if tool == "shell":
            cmd = args.get("cmd", "")
            if not cmd: return {"ok": False, "out": "missing 'cmd'"}
            lowered = cmd.lower()
            for ban in ("pip ", "pip3 ", "git ", "curl ", "wget ", "nc ", "ssh "):
                if ban in lowered:
                    return {"ok": False, "out": f"REFUSED: command includes '{ban.strip()}'"}
            rc, out = sh(cmd, cwd=wd)
            return {"ok": rc == 0, "rc": rc, "out": out}
        if tool == "read_file":
            path = wd / args.get("path", "")
            if not path.exists() or not path.resolve().is_relative_to(wd.resolve()):
                return {"ok": False, "out": "not found or outside workspace"}
            return {"ok": True, "out": path.read_text()[:4000]}
        if tool == "write_file":
            path = wd / args.get("path", "")
            try: path.resolve().relative_to(wd.resolve())
            except Exception: return {"ok": False, "out": "outside workspace"}
            path.parent.mkdir(parents=True, exist_ok=True)
            content = args.get("content", "")
            path.write_text(content)
            return {"ok": True, "out": f"wrote {len(content)} bytes to {args.get('path','')}"}
        if tool == "grep":
            pat = args.get("pattern","")
            if not pat: return {"ok": False, "out": "missing 'pattern'"}
            rc, out = sh(f"grep -nI --color=never -r {shlex.quote(pat)} .", cwd=wd)
            return {"ok": True, "rc": rc, "out": out}
        if tool == "finish":
            return {"ok": True, "out": "noted", "finish": True}
        return {"ok": False, "out": f"unknown tool {tool!r}"}
    except Exception as e:
        return {"ok": False, "out": f"tool error: {e}"}

def oracle(task, wd):
    env = os.environ.copy(); env.update(task.get("env_for_tests", {}))
    rc_grep, out_grep = sh(f"grep -r {shlex.quote(task['old_key'])} .", cwd=wd)
    old_key_hits = 0 if rc_grep == 1 else out_grep.count("\n")
    idiom_hits = 0
    for pat in task["required_idiom_patterns"]:
        rc2, _ = sh(f"grep -r {shlex.quote(pat)} .", cwd=wd)
        if rc2 == 0: idiom_hits += 1
    rc_compile, out_compile = sh("python3 -m compileall -q .", cwd=wd)
    rc_pytest, out_pytest = sh("python3 -m pytest -q --no-header --tb=short -W ignore::DeprecationWarning", cwd=wd, env=env, timeout=30)
    return {
        "old_key_hits": old_key_hits,
        "idiom_present": idiom_hits > 0,
        "compiles": rc_compile == 0,
        "pytest_rc": rc_pytest,
        "resolved": old_key_hits == 0 and idiom_hits > 0 and rc_compile == 0 and rc_pytest == 0,
        "grep_tail": out_grep[-300:],
        "pytest_tail": out_pytest[-300:],
    }

def make_system_prompt_raw(task):
    return (
        "You modify a small Python repository to replace a hardcoded API key "
        f"({task['old_key']!r}) with `os.environ[\"API_KEY\"]`.\n\n"
        "Emit ONE tool call per message on a single line beginning with `TOOL:` "
        "followed by a JSON object with keys `tool` and `args`.\n\n"
        "Tools:\n"
        "- {\"tool\": \"shell\", \"args\": {\"cmd\": \"...\"}}\n"
        "- {\"tool\": \"read_file\", \"args\": {\"path\": \"src/x.py\"}}\n"
        "- {\"tool\": \"write_file\", \"args\": {\"path\": \"src/x.py\", \"content\": \"...\"}}\n"
        "- {\"tool\": \"grep\", \"args\": {\"pattern\": \"...\"}}\n"
        "- {\"tool\": \"finish\", \"args\": {}}\n\n"
        "Forbidden: pip, git, curl, wget, ssh, nc. 25 turns."
    )

def make_system_prompt_constrained(task):
    # When constrained, we don't need a TOOL: prefix - the entire turn is a JSON tool call
    return (
        "You modify a small Python repository to replace a hardcoded API key "
        f"({task['old_key']!r}) with `os.environ[\"API_KEY\"]`.\n\n"
        "Each of your turns MUST be a single JSON object with keys `tool` and `args`.\n"
        "Valid tool names: shell, read_file, write_file, grep, finish.\n"
        "Args keys: shell->{cmd}, read_file->{path}, write_file->{path,content}, "
        "grep->{pattern}, finish->{}.\n\n"
        "Forbidden: pip, git, curl, wget, ssh, nc. 25 turns."
    )

def run_one(task, workspace_root, lane, seed, gen_fn_raw, gen_fn_constrained, fh):
    wd = workspace_root / f"attempt_{lane}_{seed}"
    seed_workspace(task, wd)

    sys_prompt = make_system_prompt_raw(task) if lane == "raw" else make_system_prompt_constrained(task)
    turns = []
    total_toks = 0
    total_gen_s = 0.0
    tool_counts = {}
    parse_failures = 0
    finished_flag = False
    t_start = time.time()

    for turn in range(task["turn_cap"]):
        msgs = [{"role":"system","content":sys_prompt}]
        if not turns:
            msgs.append({"role":"user","content":"Begin."})
        for t in turns:
            msgs.append({"role":"assistant","content":t["model"]})
            msgs.append({"role":"user","content":f"OBSERVATION:\n{json.dumps(t['obs'])[:1800]}"})

        t0 = time.time()
        gen_fn = gen_fn_raw if lane == "raw" else gen_fn_constrained
        out_text, ntok = gen_fn(msgs, seed + turn)
        total_gen_s += time.time() - t0; total_toks += ntok

        # Parse
        call = None
        parse_error = None
        if lane == "raw":
            m = TOOL_LINE_RE.search(out_text)
            if not m:
                parse_error = "no TOOL: line"
            else:
                try:
                    obj = json.loads(m.group(1))
                    if not isinstance(obj, dict) or "tool" not in obj:
                        parse_error = "missing tool key"
                    else:
                        call = obj
                except Exception as e:
                    parse_error = f"json: {e}"
        else:
            # Constrained lane: the full assistant turn should be a JSON object
            try:
                obj = json.loads(out_text.strip())
                if isinstance(obj, dict) and "tool" in obj:
                    call = obj
                else:
                    parse_error = "schema-violating json"
            except Exception as e:
                # fallback: try to find a {...} block
                try:
                    i = out_text.find("{"); j = out_text.rfind("}")
                    if 0 <= i < j: call = json.loads(out_text[i:j+1])
                    if not (isinstance(call, dict) and "tool" in call):
                        call = None; parse_error = f"json: {e}"
                except Exception:
                    call = None; parse_error = f"json: {e}"

        if call is None:
            parse_failures += 1
            turns.append({"model": out_text[:600], "obs": {"ok": False, "out": f"PARSE_ERROR: {parse_error}"}})
            continue

        tool = call.get("tool"); args = call.get("args", {}) or {}
        tool_counts[tool] = tool_counts.get(tool,0) + 1
        obs = exec_tool(tool, args, wd)
        turns.append({"model": out_text[:600], "obs": obs})
        if obs.get("finish"):
            finished_flag = True
            break

    verdict = oracle(task, wd)
    elapsed = time.time() - t_start
    rec = {
        "lane": lane, "seed": seed,
        "turns_used": len(turns),
        "model_finished": finished_flag,
        "parse_failures": parse_failures,
        "tool_counts": tool_counts,
        "verdict": verdict,
        "total_gen_s": round(total_gen_s, 3),
        "total_tokens": total_toks,
        "total_elapsed_s": round(elapsed, 3),
        "turns_head": [{"model_head": t["model"][:200], "obs_head": str(t["obs"])[:200]} for t in turns[:12]],
    }
    fh.write(json.dumps(rec)+"\n"); fh.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    args = ap.parse_args()

    task = json.loads(Path(args.task).read_text())
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

    def _run_job(ids, sampler, seed, filters=None):
        t0 = time.time()
        job_kwargs = dict(input_ids=ids, max_new_tokens=400, sampler=sampler, seed=seed, stop_conditions=eos)
        if filters is not None: job_kwargs["filters"] = filters
        job = Job(**job_kwargs)
        gen.enqueue(job)
        text=""; ntok=0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming":
                    text += r.get("text",""); ntok += 1
                    if time.time() - t0 > 30:
                        return text, ntok
        return text, ntok

    def gen_fn_raw(msgs, seed):
        ids = tok.hf_chat_template(msgs, add_generation_prompt=True)
        return _run_job(ids, DefaultSampler(), seed)

    def gen_fn_constrained(msgs, seed):
        # Use LLGuidanceFilter with JSON schema
        ids = tok.hf_chat_template(msgs, add_generation_prompt=True)
        filt = LLGuidanceFilter(tokenizer=tok, json_schema=TOOL_SCHEMA, eos_after_completed=True)
        return _run_job(ids, DefaultSampler(), seed, filters=[filt])

    with out_path.open("a") as fh:
        # Run both lanes with same seeds
        for lane in ("raw", "constrained"):
            for seed in range(9100, 9111):  # 11 seeds (greedy + 10 sampled -> 11 total sampled attempts for simplicity)
                run_one(task, workspace, lane, seed, gen_fn_raw, gen_fn_constrained, fh)

    print(f"done. receipts at {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
