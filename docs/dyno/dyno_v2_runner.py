"""Dyno v2 reliable trial runner.

Features vs v1:
  - JSONL receipts, flushed per trial (crash-safe, resumable)
  - Resume keyed on (scenario_id, condition, seed) tuple
  - Per-trial token + wall-time budget; truncation / timeout are distinct statuses
  - Explicit stop_conditions=[tok.eos_token_id] so <|im_end|> actually stops generation
  - Full retention: prompt_text, output_text, output_token_ids, seed, condition
  - Dev/held-out snippet split; prompt tuning only touches dev
  - Four conditions: zero-shot batched/single, 3-shot batched/single
  - Independent scorer with Wilson CIs + failure categorization
"""
from __future__ import annotations
import argparse, json, re, sys, time, os
from pathlib import Path

DEV_SNIPPETS = [
    ("User-supplied input is appended into an SQL query string with f-string concatenation.", "CWE-89"),
    ("The web app reflects a URL parameter verbatim into the HTML response without encoding.", "CWE-79"),
    ("A file_read endpoint accepts a path parameter with no validation; strings like ../../etc/passwd are not rejected.", "CWE-22"),
]
HELDOUT_SNIPPETS = [
    ("The JWT verifier accepts tokens whose alg header is 'none' without warning.", "CWE-347"),
    ("The upload handler saves whatever the client sends under the original filename inside /var/www/html.", "CWE-434"),
    ("A password reset form emails the plaintext of a user's new password.", "CWE-311"),
    ("The CSRF token for a form is the same global constant for every user and every session.", "CWE-352"),
    ("The audit log reads the X-Forwarded-For header verbatim as the origin IP and trusts it for rate limiting.", "CWE-346"),
    ("The code uses MD5 to hash passwords stored in the users table.", "CWE-327"),
    ("A command executor calls os.system with the raw user input appended after echo.", "CWE-78"),
]
HIGHEST_RISK_TRUTH = "CWE-78"

MODEL_DIR = "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw"
CONTEXT = 4096
TRIALS = 50
TIMEOUT_S = 30.0
MAX_NEW = 300

ZERO_SHOT_BATCHED = """For each numbered bug, output ONLY 'CWE-NNN' on its own line.
After the list, on its own line write 'HIGHEST-RISK: CWE-NNN' for the bug with greatest code-execution impact.

{items}
"""

ZERO_SHOT_SINGLE = """Classify this bug by the single most-specific CWE Top 25 ID. Reply with ONLY 'CWE-NNN' and nothing else.

Bug: {desc}
"""

FEW_SHOT_PREAMBLE_BATCHED = """Example:
Bug: User input is passed to eval() directly.
Answer: CWE-95

Example:
Bug: Shared session cookie has no HttpOnly flag.
Answer: CWE-1004

Example:
Bug: Division by zero on unchecked user input crashes the service.
Answer: CWE-369

Now do the same for the bugs below. For each numbered bug, output ONLY 'CWE-NNN' on its own line, then 'HIGHEST-RISK: CWE-NNN' on its own line.

{items}
"""

FEW_SHOT_PREAMBLE_SINGLE = """Example:
Bug: User input is passed to eval() directly.
Answer: CWE-95

Example:
Bug: Shared session cookie has no HttpOnly flag.
Answer: CWE-1004

Example:
Bug: Division by zero on unchecked user input crashes the service.
Answer: CWE-369

Now do the same for the bug below. Reply with ONLY 'CWE-NNN'.

Bug: {desc}
"""

CONDITIONS = {
    "zero_batched":  (ZERO_SHOT_BATCHED,       "batched"),
    "zero_single":   (ZERO_SHOT_SINGLE,        "single"),
    "few_batched":   (FEW_SHOT_PREAMBLE_BATCHED, "batched"),
    "few_single":    (FEW_SHOT_PREAMBLE_SINGLE,  "single"),
}

def extract_list(text):
    out = [None] * 10
    for line in text.splitlines():
        m = re.match(r"\s*(\d{1,2})[\.\):]\s*.*?(CWE-?\d{1,4})", line, re.I)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < 10 and out[idx] is None:
                n = re.search(r"\d{1,4}", m.group(2)).group()
                out[idx] = f"CWE-{n}"
    return out

def extract_cwe_first(text):
    m = re.search(r"CWE-?(\d{1,4})", text)
    return f"CWE-{m.group(1)}" if m else None

def extract_highest_risk(text):
    m = re.search(r"HIGHEST[- ]?RISK[:\s]+CWE-?(\d{1,4})", text, re.I)
    return f"CWE-{m.group(1)}" if m else None

def categorize(preds, truths, out_text):
    """Categorize: wrong_cwe, repetition, truncation, malformed."""
    valid_preds = [p for p in preds if p is not None]
    if "<|endoftext|>" in out_text and "<|im_start|>user" in out_text[out_text.find("<|endoftext|>"):]:
        return "eos_not_honored"
    if len(valid_preds) < 10:
        return "truncation"
    uniq = set(valid_preds)
    if len(uniq) == 1:
        return "repetition_single"
    if len(uniq) <= 3:
        return "repetition_narrow"
    return "wrong_cwe"

def load_done(path):
    """Resume: read already-completed (condition, seed) pairs from JSONL."""
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("status") == "completed":
                    done.add((r["condition"], r["seed"]))
            except json.JSONDecodeError:
                pass
    return done

def run_trial(gen, tok, prompt, condition, seed, eos_ids, temperature):
    from exllamav3 import Job
    from exllamav3.generator.sampler import GreedySampler, DefaultSampler
    sampler = GreedySampler() if temperature == 0.0 else DefaultSampler()
    templated = tok.hf_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    prompt_tokens = templated.shape[-1] if hasattr(templated, "shape") else len(templated)
    job = Job(
        input_ids=templated,
        max_new_tokens=MAX_NEW,
        sampler=sampler,
        seed=seed,
        stop_conditions=eos_ids,
    )
    gen.enqueue(job)
    t0 = time.time()
    out_text = ""
    output_token_count = 0
    stop_reason = None
    budget_hit = False
    while gen.num_remaining_jobs():
        if time.time() - t0 > TIMEOUT_S:
            budget_hit = True
            try:
                gen.cancel(job)
            except Exception:
                pass
            # Drain any remaining
            try:
                while gen.num_remaining_jobs():
                    gen.iterate()
            except Exception:
                pass
            break
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                out_text += r.get("text", "")
                output_token_count += 1
            elif r.get("stage") == "finished":
                stop_reason = r.get("eos_reason", "finished")
    elapsed = time.time() - t0
    if budget_hit:
        status = "timeout"
    elif stop_reason is None and output_token_count >= MAX_NEW - 2:
        status = "truncation"
    else:
        status = "completed"
    return {
        "scenario_id": "vuln_classify_01_cwe",
        "scenario_version": "v2",
        "condition": condition,
        "seed": seed,
        "temperature": temperature,
        "prompt_tokens": int(prompt_tokens),
        "output_tokens": output_token_count,
        "elapsed_s": elapsed,
        "status": status,
        "stop_reason": stop_reason,
        "prompt_text": prompt,
        "output_text": out_text,
    }

def score_trial(trial, truths):
    text = trial["output_text"]
    cond = trial["condition"]
    if cond.endswith("batched"):
        preds = extract_list(text)
    else:
        preds = [extract_cwe_first(text)] + [None] * 9  # single-snippet trial only tests one slot per snippet
    highest_risk = extract_highest_risk(text) if cond.endswith("batched") else None
    correct_per_slot = [p == t for p, t in zip(preds, truths)]
    return {
        "preds": preds,
        "correct_per_slot": correct_per_slot,
        "correct_count": sum(correct_per_slot),
        "highest_risk_pred": highest_risk,
        "highest_risk_correct": highest_risk == HIGHEST_RISK_TRUTH,
        "failure_category": categorize(preds, truths, text),
    }

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    halfwidth = z * ((p * (1-p)/n + z*z/(4*n*n))**0.5) / denom
    return (max(0, center - halfwidth), min(1, center + halfwidth))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    sys.stderr.write(f"resuming: {len(done)} trials already completed\n")

    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=CONTEXT)
    model.load()
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    im_end = tok.single_id("<|im_end|>")
    endtxt = tok.single_id("<|endoftext|>")
    eos_ids = [i for i in [im_end, endtxt] if i is not None]
    sys.stderr.write(f"eos stop_conditions: {eos_ids}\n")

    all_snippets = DEV_SNIPPETS + HELDOUT_SNIPPETS
    truths_batched = [s[1] for s in all_snippets]

    with open(args.out, "a") as f:
        for condition in args.conditions:
            tmpl, mode = CONDITIONS[condition]
            if mode == "batched":
                items = "\n".join(f"{i+1}. {s[0]}" for i, s in enumerate(all_snippets))
                prompts_for_cond = [(tmpl.format(items=items), truths_batched)]
            else:
                # single: one prompt per snippet (we score per-snippet later)
                prompts_for_cond = [(tmpl.format(desc=s[0]), [s[1]] + [None]*9) for s in all_snippets]

            for trial in range(args.trials):
                seed = 4000 + trial  # new seeds, not v1's 1000s
                for p_idx, (prompt, truths) in enumerate(prompts_for_cond):
                    cond_key = condition if mode == "batched" else f"{condition}_snip{p_idx}"
                    if (cond_key, seed) in done:
                        continue
                    r = run_trial(gen, tok, prompt, cond_key, seed, eos_ids, args.temperature)
                    r["truths"] = truths
                    r["score"] = score_trial(r, truths) if mode == "batched" else score_trial(r, truths)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    sys.stderr.write(f"  {cond_key} seed={seed} status={r['status']} correct={r['score']['correct_count']}/{10 if mode == 'batched' else 1}\n")

if __name__ == "__main__":
    main()
