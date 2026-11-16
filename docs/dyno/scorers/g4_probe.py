"""G4: Probe zero-accuracy categories on NEW dev examples only.

Held-out snippets REMAIN UNTOUCHED. We use additional dev-like paraphrases
for the 3 zero-accuracy categories (CWE-346, CWE-434, CWE-347) + CWE-311
(which was also near-zero), with three alternative prompt strategies:
  A. Minimal: "Classify: Bug..."
  B. Textbook preamble: "CWE-346 definition: Origin Validation Error..."
  C. Wiki-style: "Known vulnerabilities and their CWE IDs..."

50 seeds x 4 CWE categories x 3 prompts = 600 model calls.
"""
import json, re, sys, time
from pathlib import Path

import os
MODEL_DIR = os.environ.get("TF_MODEL_DIR", "/home/chris/models/Qwen3.5-2B-exl3-8.00bpw")
CONTEXT = 4096

# NEW probe phrasings for the 4 weakest categories
# NOTE: these are DEV-ONLY, intentionally different wording from the held-out snippets.
PROBES = {
    "CWE-346": [
        "An API endpoint uses the Origin header from the HTTP request to decide whether to approve a cross-origin fetch.",
        "The server trusts the Referer header to decide which internal pages are allowed.",
        "A webhook handler reads the X-Real-IP header verbatim and uses it to decide which caller is trusted.",
    ],
    "CWE-434": [
        "An image upload form stores the uploaded file under /var/www/public with the client-chosen filename and extension.",
        "The multipart upload handler writes the file into a directory served by the webserver without checking the content type.",
        "An avatar uploader writes PHP files into the webroot if the client sends them.",
    ],
    "CWE-347": [
        "A JWT verifier accepts tokens whose alg header is HS256 even when the signing key is the RS256 public key.",
        "An access-token verifier calls jwt.decode without verifying the signature.",
        "The server reads the JWT header's alg field and chooses which key to use based on it.",
    ],
    "CWE-311": [
        "A debugging endpoint returns the full database connection string in plaintext over HTTP.",
        "An error page prints the raw session cookie in the response body.",
        "An audit log writes user passwords to syslog on every login attempt.",
    ],
}

PROMPT_STYLES = {
    "minimal":   "Classify this software bug by the single most-specific CWE Top 25 ID. Reply with ONLY 'CWE-NNN'.\n\nBug: {desc}\n",
    "textbook":  ("Reference CWE definitions you should consider:\n"
                  "CWE-346 (Origin Validation Error): the product trusts the origin of an input (e.g., X-Forwarded-For header, Referer, Origin header) without validating that it was actually sent by the claimed origin.\n"
                  "CWE-434 (Unrestricted Upload of File with Dangerous Type): the product writes an uploaded file to a location where it can be interpreted as code without validating the file type.\n"
                  "CWE-347 (Improper Verification of Cryptographic Signature): the product does not verify or incorrectly verifies the cryptographic signature of data (e.g., JWT alg=none, signature not checked).\n"
                  "CWE-311 (Missing Encryption of Sensitive Data): the product stores or transmits sensitive data (e.g., passwords, session tokens, connection strings) without encryption.\n\n"
                  "Classify this bug by the single most-specific CWE from the list above. Reply with ONLY 'CWE-NNN'.\n\nBug: {desc}\n"),
    "wiki":     ("Examples of known CWE Top-25 vulnerabilities:\n"
                  "- SQL injection via string concatenation: CWE-89\n"
                  "- Reflected XSS: CWE-79\n"
                  "- Path traversal: CWE-22\n"
                  "- OS command injection via os.system with user input: CWE-78\n"
                  "- Weak password hashing with MD5: CWE-327\n\n"
                  "Classify this bug by the single most-specific CWE Top 25 ID. Reply with ONLY 'CWE-NNN'.\n\nBug: {desc}\n"),
}

def extract_cwe_first(text):
    m = re.search(r"CWE-?(\d{1,4})", text)
    return f"CWE-{m.group(1)}" if m else None

def main():
    import torch, gc
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
    from exllamav3.generator.sampler import DefaultSampler
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=CONTEXT)
    model.load()
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    im_end = tok.single_id("<|im_end|>")
    endtxt = tok.single_id("<|endoftext|>")
    eos_ids = [i for i in [im_end, endtxt] if i is not None]

    SEEDS = list(range(7000, 7050))  # 50 seeds, non-overlapping with prior

    out_path = Path(os.environ.get("TF_STAGE7J_OUT", "/home/chris/stage7j/g4_probe.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w")

    for truth_cwe, descs in PROBES.items():
        for d_idx, desc in enumerate(descs):
            for style_name, tmpl in PROMPT_STYLES.items():
                for seed in SEEDS:
                    prompt = tmpl.format(desc=desc)
                    templated = tok.hf_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)
                    job = Job(input_ids=templated, max_new_tokens=100, sampler=DefaultSampler(), seed=seed, stop_conditions=eos_ids)
                    gen.enqueue(job)
                    out_text = ""
                    while gen.num_remaining_jobs():
                        for r in gen.iterate():
                            if r.get("stage") == "streaming":
                                out_text += r.get("text","")
                    pred = extract_cwe_first(out_text)
                    correct = pred == truth_cwe
                    f.write(json.dumps({
                        "truth": truth_cwe, "desc_idx": d_idx, "style": style_name,
                        "seed": seed, "pred": pred, "correct": correct, "out_head": out_text[:100]
                    }) + "\n")
                    f.flush()
        sys.stderr.write(f"done {truth_cwe}\n")
    f.close()

    # Summary
    rows = [json.loads(l) for l in open(out_path)]
    print(f"\ntotal trials: {len(rows)}")
    from collections import defaultdict
    by_cond = defaultdict(lambda: [0,0])
    for r in rows:
        key = (r["truth"], r["style"])
        by_cond[key][1] += 1
        if r["correct"]:
            by_cond[key][0] += 1
    print(f"\n{'truth':10s} {'style':10s}  {'correct/total':>15s}  {'accuracy':>10s}")
    for (truth, style), (k, n) in sorted(by_cond.items()):
        print(f"{truth:10s} {style:10s}  {k:4d}/{n:3d}  {k*100/n:7.1f}%")

if __name__ == "__main__":
    main()
