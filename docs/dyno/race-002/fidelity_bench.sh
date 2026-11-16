#!/usr/bin/env bash
# Run 4 micro-fidelity domains (coding, reasoning, instruction-following, cybersec-classification)
# against a GGUF model via llama-server. Deterministic sampling.
set -u
LCPP="${LCPP:-/home/chris/.lmstudio/extensions/backends/llama.cpp-linux-arm64-nvidia-cuda13-2.33.0}"
export LD_LIBRARY_PATH="$LCPP:${LD_LIBRARY_PATH:-}"
MODEL="$1"; LABEL="$2"; PORT="${PORT:-18090}"
OUT_JSON="${OUT_JSON:-/tmp/fidelity_${LABEL}.json}"
STDERR="/tmp/fid_stderr_${LABEL}.txt"

"$LCPP/llama-server" -m "$MODEL" --port "$PORT" --host 127.0.0.1 -ngl 99 --ctx-size 2048 --parallel 1 --no-warmup > "$STDERR" 2>&1 &
PID=$!
for i in $(seq 1 240); do
  if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q "ok"; then break; fi
  sleep 0.5
done

# Deterministic greedy sample helper
ask() {
    local prompt="$1"; local ntok="$2"
    python3 -c '
import json, sys, urllib.request
req = json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]), "temperature": 0.0, "top_p": 1.0, "seed": 0}).encode()
r = urllib.request.urlopen(urllib.request.Request(sys.argv[3], data=req, headers={"Content-Type":"application/json"}), timeout=60)
d = json.load(r)
print(json.dumps({"content": d.get("content",""), "tokens": d.get("tokens_predicted", 0)}))
' "$prompt" "$ntok" "http://127.0.0.1:$PORT/completion"
}

# ---- Coding (5 HumanEval-style problems, check for correct expected keyword) ----
CODING_PROMPTS=(
    "Write a Python function has_close_elements(numbers, threshold) that returns True if any two numbers in the list are within threshold of each other, else False. Only output the function body, no explanation.\n\ndef has_close_elements(numbers, threshold):\n"
    "Write a Python function is_palindrome(s) that returns True if s is a palindrome (case-insensitive, ignoring non-alphanumeric). Only the function body.\n\ndef is_palindrome(s):\n"
    "Write a Python function sum_digits(n) that returns the sum of decimal digits of positive integer n. Just the function body.\n\ndef sum_digits(n):\n"
    "Write a Python function factorial(n) using iteration. Just the function body.\n\ndef factorial(n):\n"
    "Write a Python function fizzbuzz(n) that returns a list of strings for 1..n with Fizz/Buzz/FizzBuzz rules. Just the function body.\n\ndef fizzbuzz(n):\n"
)
# ---- Reasoning (5 GSM8K-style arithmetic word problems) ----
REASONING_PROMPTS=(
    "Q: A store sells apples at \$2 each and oranges at \$3 each. John buys 5 apples and 4 oranges. How much did he spend in total?\nA: Let me think step by step.\n"
    "Q: A train leaves at 2:00 PM traveling 60 mph. Another leaves at 3:00 PM from the same spot going the same direction at 90 mph. At what time do they meet?\nA: Let me think step by step.\n"
    "Q: Mary has 24 cookies. She gives 1/3 to her brother and eats 1/4 of what remains. How many cookies are left?\nA: Let me think step by step.\n"
    "Q: A rectangle has perimeter 40. If length is 3 more than width, what are the dimensions?\nA: Let me think step by step.\n"
    "Q: If 3 workers paint a house in 6 hours, how long will 4 workers take, assuming same rate?\nA: Let me think step by step.\n"
)
# ---- Instruction following (5 IFEval-style strict constraints) ----
IFEVAL_PROMPTS=(
    "Write a sentence that contains the word \"azure\" exactly twice. Output only the sentence, nothing else.\n"
    "List exactly three fruits, separated by commas. No other words. Output:\n"
    "Respond with only the word YES or only the word NO. Is the sky typically blue during the day?\n"
    "Write exactly one haiku about autumn leaves. Follow the 5-7-5 syllable structure. Output only the haiku.\n"
    "Translate the English word \"hello\" into Spanish. Output ONLY the Spanish word, nothing else.\n"
)
# ---- Cybersec classification (5 CWE-style) ----
CYBER_PROMPTS=(
    "Classify this vulnerability into one CWE category. Respond with ONLY the CWE number (e.g., CWE-79):\n\n'User input is passed into eval() without validation, allowing arbitrary code execution.'\n\nCWE: "
    "Classify this vulnerability into one CWE category. Respond with ONLY the CWE number:\n\n'A web form takes user data and inserts it directly into an SQL query via string concatenation.'\n\nCWE: "
    "Classify this vulnerability into one CWE category. Respond with ONLY the CWE number:\n\n'User-controlled input is reflected in the server's HTML response without encoding.'\n\nCWE: "
    "Classify this vulnerability into one CWE category. Respond with ONLY the CWE number:\n\n'The application compares passwords using == rather than a constant-time comparison function.'\n\nCWE: "
    "Classify this vulnerability into one CWE category. Respond with ONLY the CWE number:\n\n'An attacker can supply a crafted filename with ../ sequences to read files outside the upload directory.'\n\nCWE: "
)

# Expected answers (for scoring)
# coding: just capture output; scoring at analysis time
# reasoning: keyword/value check in analysis
# ifeval: constraint check in analysis
# cyber: expected = ("CWE-95","CWE-89","CWE-79","CWE-208","CWE-22")

RESULTS="{\"label\":\"$LABEL\",\"domains\":{"
# generic domain runner
run_domain() {
    local name="$1"; shift; local ntok="$1"; shift
    local arr_name="$1"; shift
    local prompts=("$@")
    local arr="["
    for i in "${!prompts[@]}"; do
        local p="${prompts[$i]}"
        local r=$(ask "$p" "$ntok")
        [ $i -gt 0 ] && arr="$arr,"
        arr="$arr{\"idx\":$i,\"prompt\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$p"),\"response\":$r}"
    done
    arr="$arr]"
    echo "\"$name\":$arr"
}

CODING_OUT=$(run_domain coding 160 _ "${CODING_PROMPTS[@]}")
REASONING_OUT=$(run_domain reasoning 200 _ "${REASONING_PROMPTS[@]}")
IFEVAL_OUT=$(run_domain ifeval 80 _ "${IFEVAL_PROMPTS[@]}")
CYBER_OUT=$(run_domain cyber 40 _ "${CYBER_PROMPTS[@]}")

RESULTS="$RESULTS$CODING_OUT,$REASONING_OUT,$IFEVAL_OUT,$CYBER_OUT}}"

kill -TERM "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
echo "$RESULTS" > "$OUT_JSON"
echo "wrote $OUT_JSON"
