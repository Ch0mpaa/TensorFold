"""Stage P2 - agentic track pilot (API-key replacement).

Simple ReAct loop:
  - System: describe the task + list tools
  - Each turn: model emits ONE tool call as a single-line JSON on a line starting
    with `TOOL:` (`{"tool": "shell", "args": {...}}`), followed by optional thought.
  - The harness parses the line, executes the tool, appends the observation, loops.
  - Up to 25 turns; then run the oracle.

Per attempt: fresh workspace, seeded files, isolated pytest run.
"""
import argparse, json, os, re, shlex, subprocess, sys, time, shutil
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_MODEL_DIR", "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw")

def sh(cmd, cwd=None, timeout=15, env=None):
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8","replace")[-2000:]
    except subprocess.TimeoutExpired as e:
        return -1, f"TIMEOUT {timeout}s"

def seed_workspace(task, wd):
    if wd.exists():
        sh(f"chmod -R u+w {wd} 2>/dev/null; rm -rf {wd}")
    wd.mkdir(parents=True)
    for rel, content in task["seed_repo_files"].items():
        p = wd / rel; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

TOOL_LINE_RE = re.compile(r"^TOOL:\s*(\{.*\})\s*$", re.MULTILINE)

def exec_tool(tool, args, wd):
    try:
        if tool == "shell":
            cmd = args.get("cmd", "")
            if not cmd:
                return {"ok": False, "out": "missing 'cmd'"}
            # Block banned invocations
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
            try:
                path.resolve().relative_to(wd.resolve())
            except Exception:
                return {"ok": False, "out": "outside workspace"}
            path.parent.mkdir(parents=True, exist_ok=True)
            content = args.get("content", "")
            path.write_text(content)
            return {"ok": True, "out": f"wrote {len(content)} bytes to {args.get('path','')}"}
        if tool == "grep":
            pat = args.get("pattern","")
            if not pat:
                return {"ok": False, "out": "missing 'pattern'"}
            rc, out = sh(f"grep -nI --color=never -r {shlex.quote(pat)} .", cwd=wd)
            return {"ok": True, "rc": rc, "out": out}
        if tool == "finish":
            return {"ok": True, "out": "noted", "finish": True}
        return {"ok": False, "out": f"unknown tool {tool!r}"}
    except Exception as e:
        return {"ok": False, "out": f"tool error: {e}"}

def oracle(task, wd):
    env = os.environ.copy()
    env.update(task.get("env_for_tests", {}))
    rc_grep, out_grep = sh(f"grep -r {shlex.quote(task['old_key'])} .", cwd=wd)
    # grep returns 1 when no matches = success
    old_key_hits = 0 if rc_grep == 1 else out_grep.count("\n")
    idiom_hits = 0
    for pat in task["required_idiom_patterns"]:
        rc2, out2 = sh(f"grep -r {shlex.quote(pat)} .", cwd=wd)
        if rc2 == 0:
            idiom_hits += 1
    rc_compile, out_compile = sh("python3 -m compileall -q .", cwd=wd)
    rc_pytest, out_pytest = sh("python3 -m pytest -q --no-header --tb=short -W ignore::DeprecationWarning", cwd=wd, env=env, timeout=30)
    verdict = {
        "old_key_hits": old_key_hits,
        "idiom_present": idiom_hits > 0,
        "compiles": rc_compile == 0,
        "pytest_rc": rc_pytest,
        "resolved": old_key_hits == 0 and idiom_hits > 0 and rc_compile == 0 and rc_pytest == 0,
        "grep_tail": out_grep[-400:],
        "compile_tail": out_compile[-400:],
        "pytest_tail": out_pytest[-400:],
    }
    return verdict

def make_system_prompt(task):
    return (
        "You modify a small Python repository to replace a hardcoded API key "
        f"({task['old_key']!r}) with `os.environ[\"API_KEY\"]`.\n\n"
        "The repository must still be importable, syntax-clean, and `pytest -q` must pass.\n\n"
        "You have tools. Emit ONE tool call per message on a single line beginning with `TOOL:` "
        "followed by a JSON object with keys `tool` and `args`. No other text may precede the TOOL line.\n\n"
        "Tools:\n"
        "- {\"tool\": \"shell\",      \"args\": {\"cmd\": \"...\"}}        — run a shell command\n"
        "- {\"tool\": \"read_file\",  \"args\": {\"path\": \"src/x.py\"}}\n"
        "- {\"tool\": \"write_file\", \"args\": {\"path\": \"src/x.py\", \"content\": \"...\"}}\n"
        "- {\"tool\": \"grep\",       \"args\": {\"pattern\": \"...\"}}\n"
        "- {\"tool\": \"finish\",     \"args\": {}}                           — you believe you are done\n\n"
        "Forbidden: pip, git, curl, wget, ssh, nc (no network). You have 25 turns.\n\n"
        "Start by exploring the repository, then make the required edits."
    )

def run_one(task, task_workspace_root, model_dir, mode, seed, gen_fn, fh):
    wd = task_workspace_root / f"attempt_{mode}_{seed}"
    seed_workspace(task, wd)

    sys_prompt = make_system_prompt(task)
    turns = []
    total_toks = 0
    total_gen_s = 0.0
    tool_counts = {}
    finished_flag = False
    t_start = time.time()

    for turn in range(task["turn_cap"]):
        msgs = [{"role":"system","content":sys_prompt}]
        if not turns:
            msgs.append({"role":"user","content":"Begin. Emit one TOOL: line now."})
        for t in turns:
            msgs.append({"role":"assistant","content":t["model"]})
            msgs.append({"role":"user","content":f"OBSERVATION:\n{json.dumps(t['obs'])[:1800]}"})
        t0 = time.time()
        out_text, ntok = gen_fn(msgs, mode, seed + turn)
        total_gen_s += time.time() - t0; total_toks += ntok

        m = TOOL_LINE_RE.search(out_text)
        if not m:
            turns.append({"model": out_text[:1200], "obs": {"ok": False, "out": "PARSE_ERROR: no TOOL: line"}})
            continue
        try:
            call = json.loads(m.group(1))
            tool = call.get("tool"); args = call.get("args", {}) or {}
        except Exception as e:
            turns.append({"model": out_text[:1200], "obs": {"ok": False, "out": f"PARSE_ERROR: {e}"}})
            continue
        tool_counts[tool] = tool_counts.get(tool,0) + 1
        obs = exec_tool(tool, args, wd)
        turns.append({"model": out_text[:1200], "obs": obs})
        if obs.get("finish"):
            finished_flag = True
            break

    # Oracle
    verdict = oracle(task, wd)
    elapsed = time.time() - t_start
    rec = {
        "mode": mode, "seed": seed,
        "turns_used": len(turns),
        "model_finished": finished_flag,
        "tool_counts": tool_counts,
        "verdict": verdict,
        "total_gen_s": round(total_gen_s, 3),
        "total_tokens": total_toks,
        "total_elapsed_s": round(elapsed, 3),
        "turns_head": [{"tool_call": t["model"][:200], "obs_head": str(t["obs"])[:200]} for t in turns[:12]],
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
    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
    cache = Cache(model, max_num_tokens=8192); model.load()
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    IM_END = tok.single_id("<|im_end|>"); ENDTXT = tok.single_id("<|endoftext|>")
    eos = [i for i in [IM_END, ENDTXT] if i is not None]

    def gen_fn(msgs, mode, seed):
        sampler = GreedySampler() if mode == "greedy" else DefaultSampler()
        ids = tok.hf_chat_template(msgs, add_generation_prompt=True)
        t0 = time.time()
        job = Job(input_ids=ids, max_new_tokens=400, sampler=sampler, seed=seed, stop_conditions=eos)
        gen.enqueue(job)
        text=""; ntok=0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming":
                    text += r.get("text",""); ntok += 1
                    if time.time() - t0 > 30:
                        return text, ntok
        return text, ntok

    with out_path.open("a") as fh:
        run_one(task, workspace, args.model_dir, "greedy", 9000, gen_fn, fh)
        for s in range(9001, 9011):
            run_one(task, workspace, args.model_dir, "sampled", s, gen_fn, fh)

    print(f"done. receipts at {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
