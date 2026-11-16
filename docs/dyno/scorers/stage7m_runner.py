"""Stage 7M runner: condition A = Stage 7L catalog (unchanged),
condition B = revised catalog (CWE-78 and CWE-89 rewritten).
Reuses the Stage 7L runner contract and the Stage 7K baseline few_single template.
"""
import argparse, json, os, sys, time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_MODEL_DIR", "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw")
BASELINE_TEMPLATE = (
    "You classify a single vulnerable code snippet by its most-specific CWE Top-25 identifier.\n\n"
    "Examples:\n\n"
    "Snippet: user_input is concatenated into an SQL string: cur.execute('SELECT * FROM x WHERE y=' + user_input)\n"
    "Answer: CWE-89\n\n"
    "Snippet: an HTTP handler echoes request.args['q'] directly into an HTML response without escaping.\n"
    "Answer: CWE-79\n\n"
    "Snippet: the server reads filename = request.args['f']; open(os.path.join('/data', filename)) without checking for '..'.\n"
    "Answer: CWE-22\n\n"
    "Now classify this snippet. Reply with ONLY 'CWE-NNN' and nothing else.\n\n"
    "Snippet (lang={lang}):\n"
    "{desc}\n"
    "Code: {code}\n\n"
    "Answer:"
)

def load_json(p): return json.loads(Path(p).read_text())
def load_body(p):
    txt = Path(p).read_text()
    i = txt.find("\n---\n")
    return txt[i+5:].strip() if i >= 0 else txt.strip()

def already_done(path):
    if not path.exists(): return set()
    seen = set()
    for line in path.open():
        try:
            r = json.loads(line); seen.add((r["condition"], r["snippet_id"], r["mode"], r["seed"]))
        except Exception: pass
    return seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets", required=True)
    ap.add_argument("--freeze",   required=True)
    ap.add_argument("--catalog-a", required=True, help="Stage 7L catalog (original)")
    ap.add_argument("--catalog-b", required=True, help="Stage 7M catalog (revised)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    args = ap.parse_args()

    snippets = load_json(args.snippets)["snippets"]
    freeze = load_json(args.freeze)
    cat_a = load_body(args.catalog_a)
    cat_b = load_body(args.catalog_b)
    eos_ids = freeze["model"]["eos_ids"]
    max_new = freeze["limits"]["max_new_tokens_per_trial"]
    timeout_s = freeze["limits"]["per_trial_timeout_seconds"]
    sampled_seeds = list(range(7000, 7050))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "deterministic.jsonl"
    smp_path = out_dir / "sampled.jsonl"
    det_done = already_done(det_path)
    smp_done = already_done(smp_path)
    print(f"resume: {len(det_done)} deterministic, {len(smp_done)} sampled already recorded", file=sys.stderr)

    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
    from exllamav3.generator.sampler import DefaultSampler, GreedySampler
    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
    cache = Cache(model, max_num_tokens=freeze["model"]["context_tokens"])
    model.load(); gen = Generator(model=model, cache=cache, tokenizer=tok)
    det_sampler = GreedySampler(); smp_sampler = DefaultSampler()

    def build_prompt(cond, s):
        base = BASELINE_TEMPLATE.format(lang=s["lang"], desc=s["desc"], code=s["code"])
        if cond == "A":
            return f"CWE reference catalog (authoritative):\n\n{cat_a}\n\n---\n\n{base}"
        if cond == "B":
            return f"CWE reference catalog (authoritative):\n\n{cat_b}\n\n---\n\n{base}"
        raise ValueError(cond)

    def run_one(cond, s, mode, seed, sampler, fh):
        key = (cond, s["id"], mode, seed)
        done = det_done if mode == "deterministic" else smp_done
        if key in done: return
        prompt = build_prompt(cond, s)
        ids = tok.hf_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)
        # Measure prompt tokens correctly this time: if ids is a tensor, use shape[-1]
        try:
            prompt_tok = int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)
        except Exception:
            prompt_tok = -1
        t0 = time.time(); t_first = None
        try:
            job = Job(input_ids=ids, max_new_tokens=max_new, sampler=sampler, seed=seed, stop_conditions=eos_ids)
            gen.enqueue(job)
            text = ""; ntok = 0
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("stage") == "streaming":
                        if t_first is None and r.get("text"): t_first = time.time()
                        text += r.get("text",""); ntok += 1
                if time.time() - t0 > timeout_s:
                    status = "timeout"; break
            else:
                status = "completed"
        except Exception as e:
            text = ""; ntok = 0; status = f"failure:{e}"
        rec = {
            "condition": cond, "snippet_id": s["id"], "cwe_truth": s["cwe"],
            "mode": mode, "seed": seed, "status": status,
            "elapsed_s": round(time.time() - t0, 4),
            "ttft_s": round(t_first - t0, 4) if t_first else None,
            "prompt_tokens": prompt_tok, "n_tokens": ntok,
            "output_text": text,
        }
        fh.write(json.dumps(rec) + "\n"); fh.flush()

    with det_path.open("a") as fh:
        for cond in ("A","B"):
            for s in snippets:
                run_one(cond, s, "deterministic", 0, det_sampler, fh)
    with smp_path.open("a") as fh:
        for cond in ("A","B"):
            for s in snippets:
                for seed in sampled_seeds:
                    run_one(cond, s, "sampled", seed, smp_sampler, fh)

    print(f"done. {det_path} + {smp_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
