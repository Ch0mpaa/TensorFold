"""P2c intervention: grep-before-finish gate.

MINIMAL intervention (ONE rule change):
  - Model's finish tool call is REJECTED by the harness unless the model
    has, at some point in the trace, called `grep` with a pattern matching
    the task's `old_key` AND that grep returned 0 matches (rc=1).
  - On rejection, harness returns a REJECTED observation describing the
    rule and returns the model to its loop. The 25-turn cap is UNCHANGED.

Everything else (schema, tool set, oracle, task files) is byte-identical to
harness_diag.py. The intervention is a SINGLE if-branch in exec_tool's finish
handler. No prompt change; the harness emits the rejection as an OBSERVATION
so the model learns the rule in context.

Scoring: same oracle. Paired with harness_diag.py on identical seeds.
"""
import argparse, json, os, re, shlex, subprocess, sys, time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_MODEL_DIR", "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw")

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

class FinishGate:
    """Tracks whether a grep for old_key has ever returned zero matches."""
    def __init__(self, old_key):
        self.old_key = old_key
        self.saw_zero_grep = False
    def observe_grep(self, pattern, rc, out):
        # count old_key occurrences in the returned matches
        if self.old_key in pattern or pattern in self.old_key:
            # rc==1 from grep means "no matches"
            if rc == 1:
                self.saw_zero_grep = True

def exec_tool(tool, args, wd, gate: FinishGate):
    try:
        if tool == "shell":
            cmd = args.get("cmd", "")
            if not cmd: return {"ok": False, "out": "missing 'cmd'"}
            lowered = cmd.lower()
            for ban in ("pip ", "pip3 ", "git ", "curl ", "wget ", "nc ", "ssh ", "rm -rf", "rm -r", "rm ."):
                if ban in lowered: return {"ok": False, "out": f"REFUSED: command includes '{ban.strip()}'"}
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
            gate.observe_grep(pat, rc, out)
            return {"ok": True, "rc": rc, "out": out}
        if tool == "finish":
            # INTERVENTION: reject finish until gate.saw_zero_grep
            if not gate.saw_zero_grep:
                return {"ok": False, "out": (
                    f"REJECTED: before you may call finish you must run "
                    f"grep with pattern '{gate.old_key}' (or a substring) and "
                    f"receive rc=1 (no matches). Verify your edit removed every "
                    f"occurrence, then try finish again."
                )}
            return {"ok": True, "out": "noted", "finish": True}
        return {"ok": False, "out": f"unknown tool {tool!r}"}
    except Exception as e:
        return {"ok": False, "out": f"tool error: {e}"}

def oracle(task, wd):
    if not Path(wd).exists():
        return {"old_key_hits": -1, "idiom_present": False, "compiles": False,
                "pytest_rc": -1, "resolved": False, "pytest_tail": "WORKSPACE_DELETED"}
    env = os.environ.copy(); env.update(task.get("env_for_tests", {}))
    rc_grep, out_grep = sh(f"grep -r {shlex.quote(task['old_key'])} .", cwd=wd)
    old_key_hits = 0 if rc_grep == 1 else out_grep.count("\n")
    idiom_hits = 0
    for pat in task["required_idiom_patterns"]:
        rc2, _ = sh(f"grep -r {shlex.quote(pat)} .", cwd=wd)
        if rc2 == 0: idiom_hits += 1
    rc_compile, _ = sh("python3 -m compileall -q .", cwd=wd)
    rc_pytest, out_pytest = sh("python3 -m pytest -q --no-header --tb=short -W ignore::DeprecationWarning", cwd=wd, env=env, timeout=30)
    return {
        "old_key_hits": old_key_hits,
        "idiom_present": idiom_hits > 0,
        "compiles": rc_compile == 0,
        "pytest_rc": rc_pytest,
        "resolved": old_key_hits == 0 and idiom_hits > 0 and rc_compile == 0 and rc_pytest == 0,
        "pytest_tail": out_pytest[-200:],
    }

def run_one(task, workspace_root, seed, gen_fn, fh):
    wd = workspace_root / f"attempt_intervened_{seed}"
    seed_workspace(task, wd)
    gate = FinishGate(task["old_key"])

    is_api_key = 'api' in task.get('task_id','').lower()
    sys_prompt = (
        "You modify a small Python repository to satisfy the task below.\n\n"
        f"TASK: Replace the hardcoded value `{task['old_key']}` throughout the repo.\n"
        + (f"Replace it with `os.environ[\"API_KEY\"]`.\n" if is_api_key else "Remove every matching line entirely.\n") +
        "Each of your turns MUST be a single JSON object with keys `tool` and `args`.\n"
        "Valid tool names: shell, read_file, write_file, grep, finish.\n"
        "Args keys: shell->{cmd}, read_file->{path}, write_file->{path,content}, "
        "grep->{pattern}, finish->{}. Forbidden: pip, git, curl, wget, ssh, nc. 25 turns.\n\n"
        "NOTE: the harness will REJECT any finish call until it observes a grep for "
        f"pattern `{task['old_key']}` that returns zero matches. Verify before you finish."
    )
    turns = []
    total_toks = 0
    finished_flag = False
    t_start = time.time()
    finish_rejections = 0

    for turn in range(task["turn_cap"]):
        msgs = [{"role":"system","content":sys_prompt}]
        if not turns:
            msgs.append({"role":"user","content":"Begin."})
        for t in turns:
            msgs.append({"role":"assistant","content":t["model"]})
            msgs.append({"role":"user","content":f"OBSERVATION:\n{json.dumps(t['obs'])[:1800]}"})
        out_text, ntok = gen_fn(msgs, seed + turn)
        total_toks += ntok
        try:
            obj = json.loads(out_text.strip())
            call = obj if isinstance(obj, dict) and "tool" in obj else None
        except Exception:
            call = None
            try:
                i = out_text.find("{"); j = out_text.rfind("}")
                if 0 <= i < j: call = json.loads(out_text[i:j+1])
                if not (isinstance(call, dict) and "tool" in call): call = None
            except Exception: call = None
        if call is None:
            turns.append({"model": out_text[:400], "call": None, "obs": {"ok": False, "out": "PARSE_ERROR"}})
            continue
        tool = call.get("tool"); args = call.get("args", {}) or {}
        obs = exec_tool(tool, args, wd, gate)
        if tool == "finish" and not obs.get("finish"):
            finish_rejections += 1
        turns.append({"model": out_text[:400], "call": call, "obs": obs})
        if obs.get("finish"):
            finished_flag = True; break

    verdict = oracle(task, wd)
    rec = {
        "seed": seed, "turns_used": len(turns),
        "model_finished": finished_flag,
        "finish_rejections": finish_rejections,
        "gate_satisfied_before_finish": gate.saw_zero_grep,
        "verdict": verdict,
        "total_tokens": total_toks,
        "total_elapsed_s": round(time.time() - t_start, 3),
        "full_turns": [{"call": t.get("call"), "obs": t.get("obs")} for t in turns],
    }
    fh.write(json.dumps(rec)+"\n"); fh.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--seed-start", type=int, default=9100)
    ap.add_argument("--n-seeds", type=int, default=11)
    args = ap.parse_args()

    task = json.loads(Path(args.task).read_text())
    workspace = Path(args.workspace); workspace.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
    from exllamav3.generator.sampler import DefaultSampler
    from exllamav3.generator.filter import LLGuidanceFilter
    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
    cache = Cache(model, max_num_tokens=8192); model.load()
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    IM_END = tok.single_id("<|im_end|>"); ENDTXT = tok.single_id("<|endoftext|>")
    eos = [i for i in [IM_END, ENDTXT] if i is not None]

    SCHEMA = {
        "type":"object","additionalProperties":False,"required":["tool","args"],
        "properties":{
            "tool":{"type":"string","enum":["shell","read_file","write_file","grep","finish"]},
            "args":{"type":"object","additionalProperties":True}
        }
    }
    def gen_fn(msgs, seed):
        ids = tok.hf_chat_template(msgs, add_generation_prompt=True)
        t0 = time.time()
        filt = LLGuidanceFilter(tokenizer=tok, json_schema=SCHEMA, eos_after_completed=True)
        job = Job(input_ids=ids, max_new_tokens=400, sampler=DefaultSampler(), seed=seed, stop_conditions=eos, filters=[filt])
        gen.enqueue(job)
        text=""; ntok=0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming":
                    text += r.get("text",""); ntok += 1
                    if time.time() - t0 > 30: return text, ntok
        return text, ntok

    with out_path.open("a") as fh:
        for i in range(args.n_seeds):
            run_one(task, workspace, args.seed_start + i, gen_fn, fh)

    print(f"done. receipts at {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
