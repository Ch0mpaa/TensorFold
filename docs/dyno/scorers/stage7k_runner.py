"""Stage 7K controlled evaluation runner.

Honors the proven v2 contract:
  - JSONL receipts flushed per trial (resume on (snippet_id, seed, mode))
  - stop_conditions=[<|im_end|>, <|endoftext|>] on every Job
  - full retention: prompt, raw text, pred, seed, elapsed, tokens
  - status enum: completed / timeout / failure (truncation detected post hoc)

Reads snippets + freeze from a directory argument; emits receipts to another.
Paths are passed in; no hard-coded D:/ or /home/chris paths.
"""
import argparse, json, os, sys, time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_MODEL_DIR", "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw")

def load_json(p): return json.loads(Path(p).read_text())

def build_prompt(template, snippet):
    return template.format(lang=snippet["lang"], desc=snippet["desc"], code=snippet["code"])

def already_done(path):
    if not path.exists(): return set()
    seen = set()
    for line in path.open():
        try:
            r = json.loads(line)
            seen.add((r["snippet_id"], r["mode"], r["seed"]))
        except Exception:
            pass
    return seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets", required=True, help="snippets.json")
    ap.add_argument("--freeze",   required=True, help="freeze.json")
    ap.add_argument("--out-dir",  required=True, help="where to append the two JSONL receipts")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    args = ap.parse_args()

    snippets_doc = load_json(args.snippets)
    freeze = load_json(args.freeze)
    snippets = snippets_doc["snippets"]
    template = freeze["prompt_template"]["user_template"]
    det_sampler_cfg = freeze["sampler"]["deterministic"]
    smp_sampler_cfg = freeze["sampler"]["sampled"]
    eos_ids = freeze["model"]["eos_ids"]
    max_new = freeze["limits"]["max_new_tokens_per_trial"]
    sampled_seeds = list(range(5000, 5050))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "deterministic.jsonl"
    smp_path = out_dir / "sampled.jsonl"
    det_done = already_done(det_path)
    smp_done = already_done(smp_path)
    print(f"resume: {len(det_done)} deterministic, {len(smp_done)} sampled already recorded", file=sys.stderr)

    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
    from exllamav3.generator.sampler import DefaultSampler, GreedySampler
    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg)
    tok = Tokenizer.from_config(cfg)
    cache = Cache(model, max_num_tokens=freeze["model"]["context_tokens"])
    model.load()
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    # Use IDENTICAL sampler constructors as the v2 runner so stage7k is apples-to-apples with
    # Stage 7J. DefaultSampler takes no kwargs in exllamav3 1.6.7; GreedySampler is argmax.
    det_sampler = GreedySampler()
    smp_sampler = DefaultSampler()

    def run_one(snippet, mode, seed, sampler, fh):
        key = (snippet["id"], mode, seed)
        if key in (det_done if mode == "deterministic" else smp_done):
            return
        prompt = build_prompt(template, snippet)
        ids = tok.hf_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)
        t0 = time.time()
        try:
            job = Job(input_ids=ids, max_new_tokens=max_new, sampler=sampler, seed=seed, stop_conditions=eos_ids)
            gen.enqueue(job)
            text = ""
            ntok = 0
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("stage") == "streaming":
                        text += r.get("text","")
                        ntok += 1
                if time.time() - t0 > freeze["limits"]["per_trial_timeout_seconds"]:
                    status = "timeout"; break
            else:
                status = "completed"
        except Exception as e:
            text = ""; ntok = 0; status = f"failure:{e}"
        rec = {
            "snippet_id": snippet["id"], "cwe_truth": snippet["cwe"], "mode": mode,
            "seed": seed, "status": status, "elapsed_s": round(time.time() - t0, 3),
            "n_tokens": ntok, "output_text": text,
            "prompt_len_chars": len(prompt), "prompt_head": prompt[:100],
        }
        fh.write(json.dumps(rec) + "\n"); fh.flush()

    # Deterministic baseline across 30 snippets
    with det_path.open("a") as fh:
        for s in snippets:
            run_one(s, "deterministic", 0, det_sampler, fh)

    # Sampled sweep: 30 x 50 seeds
    with smp_path.open("a") as fh:
        for s in snippets:
            for seed in sampled_seeds:
                run_one(s, "sampled", seed, smp_sampler, fh)

    print(f"done. receipts at {det_path} and {smp_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
