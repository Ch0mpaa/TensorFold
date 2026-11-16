"""P1 fresh held-out: self-contained synthetic instance to test generalization
of the raw-vs-assisted lane effect beyond pallets__flask-5014.

NOT from the SWE-bench Verified manifest. Transparently labeled as a
2b-authored repro. Same lane/scoring/oracle shape as p1b/runner_both.py.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_MODEL_DIR", "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "p1b"))
from runner_both import (extract_patch, extract_edit_spec, apply_edit_spec,
                         make_raw_prompt, make_assisted_prompt, sh)

def run_one(instance, workspace_root, repo_base, lane, seed, sampler_name, gen_fn, fh):
    attempt_dir = workspace_root / f"attempt_{lane}_{sampler_name}_{seed}"
    if attempt_dir.exists():
        sh(f"chmod -R u+w {attempt_dir} 2>/dev/null; rm -rf {attempt_dir}")

    sh(f"git clone --quiet {repo_base} {attempt_dir}")
    # HEAD is the base state (fresh repo with bug). No test_patch to apply.

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
        "gen_elapsed_s": round(gen_elapsed, 3), "ttft_s": ttft, "n_tokens": ntok,
        "prompt_chars": len(prompt), "raw_output_head": out_text[:800],
    }

    if lane == "raw":
        patch = extract_patch(out_text)
        rec["patch_extracted"] = patch is not None
        if patch is None:
            rec["status"] = "malformed_patch"; fh.write(json.dumps(rec)+"\n"); fh.flush(); return
        with (attempt_dir/"_model_patch.diff").open("w") as g: g.write(patch)
        rc, out = sh("git apply --check _model_patch.diff", cwd=attempt_dir)
        rec["patch_valid_check"] = rc == 0; rec["check_out_head"] = out[:300]
        if rc != 0:
            rec["status"] = "patch_invalid_check"; fh.write(json.dumps(rec)+"\n"); fh.flush(); return
        rc, _ = sh("git apply --index _model_patch.diff", cwd=attempt_dir)
        rec["patch_apply_rc"] = rc
        if rc != 0:
            rec["status"] = "patch_apply_failed"; fh.write(json.dumps(rec)+"\n"); fh.flush(); return
    else:
        spec = extract_edit_spec(out_text)
        rec["spec_extracted"] = spec is not None
        if spec is None:
            rec["status"] = "malformed_spec"; fh.write(json.dumps(rec)+"\n"); fh.flush(); return
        rec["spec_head"] = {k: (v[:200] if isinstance(v,str) else v) for k,v in spec.items()}
        ok, reason = apply_edit_spec(spec, attempt_dir, instance)
        rec["spec_apply_ok"] = ok; rec["spec_apply_reason"] = reason
        if not ok:
            rec["status"] = f"spec_apply_failed:{reason}"; fh.write(json.dumps(rec)+"\n"); fh.flush(); return
        rc, patch_body = sh("git diff --no-color HEAD", cwd=attempt_dir)
        rec["generated_patch_bytes"] = len(patch_body)
        with (attempt_dir/"_generated_patch.diff").open("w") as g: g.write(patch_body)
        sh(f"git checkout {instance['target_file']}", cwd=attempt_dir)
        rc_chk, out_chk = sh("git apply --check _generated_patch.diff", cwd=attempt_dir)
        rec["patch_valid_check"] = rc_chk == 0; rec["check_out_head"] = out_chk[:300]
        if rc_chk != 0:
            rec["status"] = "generated_patch_invalid"; fh.write(json.dumps(rec)+"\n"); fh.flush(); return
        rc_app, _ = sh("git apply --index _generated_patch.diff", cwd=attempt_dir)
        rec["patch_apply_rc"] = rc_app
        if rc_app != 0:
            rec["status"] = "generated_patch_apply_failed"; fh.write(json.dumps(rec)+"\n"); fh.flush(); return

    # oracle: just run pytest; no separate FAIL_TO_PASS/PASS_TO_PASS split
    env = os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"] = "1"
    t1 = time.time()
    rc_pytest, out_pytest = sh("python3 -m pytest -q --no-header --tb=no tests/", cwd=attempt_dir, timeout=30, env=env)
    rec["pytest_elapsed_s"] = round(time.time()-t1, 3)
    rec["pytest_rc"] = rc_pytest
    rec["pytest_out_tail"] = out_pytest[-400:]
    rec["resolved"] = rc_pytest == 0
    rec["status"] = "resolved" if rec["resolved"] else "pytest_fail"
    fh.write(json.dumps(rec)+"\n"); fh.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--repo-base", required=True)
    ap.add_argument("--workspace", required=True)
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
        job = Job(input_ids=ids, max_new_tokens=1000, sampler=sampler, seed=seed, stop_conditions=eos)
        gen.enqueue(job)
        text=""; ntok=0
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming":
                    if t_first is None and r.get("text"): t_first = time.time()
                    text += r.get("text",""); ntok += 1
                    if time.time() - t0 > 120:
                        return text, ntok, round(t_first-t0,3) if t_first else None
        return text, ntok, round(t_first-t0, 3) if t_first else None

    with out_path.open("a") as fh:
        for lane in ("raw", "assisted"):
            run_one(instance, workspace, args.repo_base, lane, 10000, "greedy", gen_fn, fh)
            for s in (10001, 10002, 10003):
                run_one(instance, workspace, args.repo_base, lane, s, "default", gen_fn, fh)
    print(f"done. receipts at {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
