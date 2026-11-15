"""Independent scorer v2b: no regex, pure string/list parsing.

Algorithm:
  1. Strip ChatML / endoftext markers and newlines into a token stream.
  2. Scan for "CWE" followed by optional "-" and digits; validate against
     a catalog allowlist (known CWE Top-25 plus a few neighbors).
  3. Return (ordered_list_of_cwe_ids, highest_risk_pick).
"""
import json, sys
from pathlib import Path

# CWE catalog: Top 25 + a handful of commonly cited CWE IDs so we can recognize
# wrong but validly-formatted answers (e.g. the model picks CWE-95 for an eval()
# example). The scorer only needs to extract what the model emitted; truth
# comparison happens in the caller. The allowlist lets us reject garbage like
# CWE-891 if we want to be strict; here we accept any 2-5 digit CWE-NNN.
VALID_CWE_RANGES = [(1, 20000)]

def _is_digit(ch): return ch.isdigit()

def _parse_cwe_at(text, i):
    """If text[i:] starts with 'CWE' followed by optional dash and digits,
    return (cwe_id_string, end_index). Else (None, i)."""
    if i + 3 > len(text) or text[i:i+3] != "CWE":
        return None, i
    j = i + 3
    if j < len(text) and text[j] == "-":
        j += 1
    k = j
    while k < len(text) and _is_digit(text[k]):
        k += 1
    if k == j:
        return None, i  # no digits
    num = text[j:k]
    if not (1 <= int(num) <= 20000):
        return None, i
    return f"CWE-{num}", k

def scan_cwe_tokens(text):
    """Return list of (cwe_id, start_index) in order of appearance."""
    out = []
    i = 0
    while i < len(text):
        cwe, end = _parse_cwe_at(text, i)
        if cwe:
            out.append((cwe, i))
            i = end
        else:
            i += 1
    return out

def _cleanup(text):
    """Strip trailing ChatML junk. For the batched format, we want only the FIRST
    assistant turn; anything after the first <|im_end|> is the model running past
    EOS (the pathology we fixed in v2 scorer_v2a) and must be ignored for parsing."""
    cut = text.find("<|im_end|>")
    if cut >= 0:
        return text[:cut]
    return text

def parse_batched(text, n_slots=10):
    """Extract a numbered list 1..n_slots from the model output.

    For each item line, look for 'N.' or 'N:' or 'N)' prefix, then the first
    CWE-NNN found after it on the same line.
    """
    first_turn = _cleanup(text)
    preds = [None] * n_slots
    for raw_line in first_turn.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Find leading number
        num_end = 0
        while num_end < len(line) and line[num_end].isdigit():
            num_end += 1
        if num_end == 0 or num_end >= len(line):
            continue
        try:
            slot = int(line[:num_end]) - 1
        except ValueError:
            continue
        if not (0 <= slot < n_slots):
            continue
        # Must be followed by . : or )
        if line[num_end] not in ".:)":
            continue
        rest = line[num_end+1:]
        tokens = scan_cwe_tokens(rest)
        if tokens and preds[slot] is None:
            preds[slot] = tokens[0][0]
    return preds

def parse_single(text):
    """Extract the FIRST CWE-NNN from the output (the model's answer)."""
    first_turn = _cleanup(text)
    tokens = scan_cwe_tokens(first_turn)
    return tokens[0][0] if tokens else None

def parse_highest_risk(text):
    """Find 'HIGHEST-RISK' (or variants) followed by a CWE id."""
    first_turn = _cleanup(text)
    upper = first_turn.upper()
    # Accept HIGHEST-RISK, HIGHEST RISK, HIGHEST_RISK
    for marker in ("HIGHEST-RISK", "HIGHEST RISK", "HIGHEST_RISK"):
        idx = upper.find(marker)
        if idx >= 0:
            after = first_turn[idx + len(marker):]
            tokens = scan_cwe_tokens(after)
            if tokens:
                return tokens[0][0]
    return None

# Unit tests (independent of scorer_v2a's regex tests)
UNIT_TESTS = [
    # Batched
    ("1. CWE-89\n2. CWE-79\n3. CWE-22\n", "batched", ["CWE-89", "CWE-79", "CWE-22"] + [None]*7, None),
    ("1. CWE-89\n2. CWE-79\n3. CWE-22\nHIGHEST-RISK: CWE-78\n", "batched", ["CWE-89", "CWE-79", "CWE-22"] + [None]*7, "CWE-78"),
    ("1. CWE-89\n1. CWE-78\n2. CWE-79\n", "batched", ["CWE-89", "CWE-79"] + [None]*8, None),  # first wins
    ("1) CWE89\n2: CWE-22\n", "batched", ["CWE-89", "CWE-22"] + [None]*8, None),
    ("1. CWE-89 (SQL injection)\n", "batched", ["CWE-89"] + [None]*9, None),
    ("", "batched", [None]*10, None),
    ("CWE-89 is the answer\n", "batched", [None]*10, None),  # no leading number
    ("1. CWE-89\n2. CWE-79<|im_end|>\n3. CWE-22\n", "batched", ["CWE-89", "CWE-79"] + [None]*8, None),  # cut at im_end
    ("11. CWE-89\n", "batched", [None]*10, None),  # out of range
    # Single
    ("CWE-89\n", "single", "CWE-89", None),
    ("The answer is CWE-89 I think.\n", "single", "CWE-89", None),
    ("CWE-89<|im_end|>garbage CWE-78", "single", "CWE-89", None),
    ("CWE-", "single", None, None),  # no digits
    ("CWE 89", "single", None, None),  # space between CWE and number
    ("cwe-89", "single", None, None),  # lowercase rejected
    # Highest risk formats
    ("...HIGHEST-RISK: CWE-78", "hr", None, "CWE-78"),
    ("highest risk: CWE-78", "hr", None, "CWE-78"),
    ("Highest_Risk CWE-22", "hr", None, "CWE-22"),
    ("HIGHEST-RISK\n\nCWE-22", "hr", None, "CWE-22"),
    ("...risky but no marker CWE-78", "hr", None, None),
]

def run_unit_tests():
    failures = []
    for i, (inp, mode, want_list_or_pred, want_hr) in enumerate(UNIT_TESTS):
        if mode == "batched":
            got = parse_batched(inp)
            if got != want_list_or_pred:
                failures.append((i, mode, inp, got, want_list_or_pred))
        elif mode == "single":
            got = parse_single(inp)
            if got != want_list_or_pred:
                failures.append((i, mode, inp, got, want_list_or_pred))
        elif mode == "hr":
            got = parse_highest_risk(inp)
            if got != want_hr:
                failures.append((i, mode, inp, got, want_hr))
    return failures

if __name__ == "__main__":
    failures = run_unit_tests()
    print(f"scorer_v2b unit tests: {len(UNIT_TESTS)} total, {len(failures)} failures")
    for f in failures:
        print(f"  FAIL #{f[0]} mode={f[1]} input={f[2]!r} got={f[3]!r} want={f[4]!r}")
    sys.exit(0 if not failures else 1)
