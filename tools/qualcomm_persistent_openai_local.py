#!/usr/bin/env python3
"""Phone-local OpenAI HTTP/SSE server backed by one persistent Qualcomm TensorFold worker."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib, json, os, subprocess, threading, time
from qwen35_tokenizer import Qwen35Tokenizer

MODEL_ID = os.environ.get("TF_MODEL_ID", "qwen35-4b-exl3-4bpw-local")
HTTP_HOST = os.environ.get("TF_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("TF_HTTP_PORT", "8082"))
TOKENIZER_DIR = os.environ.get("TF_TOKENIZER_DIR", "/data/local/tmp/tensorfold-models/Qwen3.5-2B-exl3-8.00bpw")
BUNDLE = os.environ.get("TF_BUNDLE", "/data/local/tmp/exl3-full-power")
BINARY = os.environ.get("TF_BINARY", BUNDLE + "/tensorfold-qualcomm-exl3-forward-4b-persistent-http-v1")
MODEL = os.environ.get("TF_MODEL", "/data/local/tmp/tensorfold-models/Qwen3.5-4B-exl3-4.00bpw/model.safetensors")
CACHE = os.environ.get("TF_CACHE", "/data/local/tmp/tensorfold-exl3-cache-4b-v1")
MTP = os.environ.get("TF_MTP", "/data/local/tmp/tensorfold-mtp-poc/mtp.safetensors")
MODEL_SHA = os.environ.get("TF_MODEL_SHA", "117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1")
MODE = os.environ.get("TF_MODE", "serve-q4-full-out-q8-q0-head-adaptive-linear")
MAX_STEPS = int(os.environ.get("TF_MAX_STEPS", "1024"))
MTP_DEPTH = int(os.environ.get("TF_MTP_DEPTH", "2"))
MTP_INITIAL_RANK = int(os.environ.get("TF_MTP_INITIAL_RANK", "1"))
MTP_FIRST_MARGIN = os.environ.get("TF_MTP_FIRST_MARGIN", "0.1").strip() or None
MTP_SECOND_MARGIN = os.environ.get("TF_MTP_SECOND_MARGIN", "").strip() or None
TARGET_AMBIGUITY_MARGIN = os.environ.get("TF_TARGET_AMBIGUITY_MARGIN", "0.125").strip() or None
Q8_OUTPUT_END = int(os.environ.get("TF_Q8_OUTPUT_END", "24"))
FUSED_LINEAR_MASK = os.environ.get("TF_FUSED_LINEAR_MASK", "0")
HMX_MIN_ROWS = int(os.environ.get("TF_HMX_MIN_ROWS", "4"))
OPPOLL = int(os.environ.get("TF_OPPOLL", "1"))
POWER_MODE = os.environ.get("TF_POWER_MODE", "low_balanced")
PROFILE_PREFILL = os.environ.get("TF_PERSISTENT_PROFILE") == "1"
TOKENIZER = Qwen35Tokenizer(TOKENIZER_DIR)
REQUEST_LOCK = threading.Lock()


def encode_request(path, body):
    if path.endswith("/chat/completions"):
        options = body.get("chat_template_kwargs") or {}
        template = TOKENIZER.chat_template(body.get("messages") or [],
                                           enable_thinking=bool(options.get("enable_thinking", False)))
        return TOKENIZER.encode(template, parse_special=True)
    return TOKENIZER.encode(body.get("prompt", ""), parse_special=True)


def decode_tokens(tokens):
    return TOKENIZER.decode(tokens)


def token_sha(tokens):
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


class PersistentWorker:
    def __init__(self):
        self.process = None
        self.pid = None
        self.ready = False
        self.next_id = 1

    def command(self):
        env = os.environ.copy()
        env.update({
            "LD_LIBRARY_PATH": BUNDLE + "/lib", "ADSP_LIBRARY_PATH": BUNDLE + "/lib",
            "GGML_HEXAGON_OPPOLL": str(OPPOLL), "GGML_HEXAGON_HMX_PIPELINE_MIN_ROWS": str(HMX_MIN_ROWS),
            "GGML_HEXAGON_POWER_MODE": POWER_MODE,
            "TENSORFOLD_MTP_SIDECAR": MTP, "TENSORFOLD_MTP_DEPTH": str(MTP_DEPTH),
            "TENSORFOLD_MTP_INITIAL_RANK": str(MTP_INITIAL_RANK),
            "TENSORFOLD_Q8_OUTPUT_LAYER_END": str(Q8_OUTPUT_END),
            "TENSORFOLD_FUSED_LINEAR_LAYER_MASK": str(FUSED_LINEAR_MASK),
            "TENSORFOLD_BENCH_QUIET": "1", "TENSORFOLD_STREAM_TOKENS": "1",
            "TENSORFOLD_PERSISTENT_WORKER": "1",
        })
        if MTP_FIRST_MARGIN is not None: env["TENSORFOLD_MTP_FIRST_MARGIN"] = MTP_FIRST_MARGIN
        if MTP_SECOND_MARGIN is not None: env["TENSORFOLD_MTP_SECOND_MARGIN"] = MTP_SECOND_MARGIN
        if TARGET_AMBIGUITY_MARGIN is not None: env["TENSORFOLD_TARGET_AMBIGUITY_MARGIN"] = TARGET_AMBIGUITY_MARGIN
        if PROFILE_PREFILL: env["TENSORFOLD_PREFILL_PROFILE"] = "1"
        args = [BINARY, "htp", MODEL, "0", "6", CACHE, MODEL_SHA, str(MAX_STEPS), "0", MODE, "1"]
        return args, env

    def start(self):
        self.stop()
        args, env = self.command()
        self.process = subprocess.Popen(args, cwd=BUNDLE, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.pid = self.process.pid
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line: break
            line = line.strip()
            if line.startswith("{"):
                try: event = json.loads(line)
                except json.JSONDecodeError: continue
                if event.get("tensorfold") == "qualcomm-exl3-worker-ready":
                    self.ready = True
                    print(f"persistent worker ready pid={self.pid}", flush=True)
                    return
            elif line:
                print(f"device startup: {line[:240]}", flush=True)
        raise RuntimeError("persistent worker exited before ready")

    def ensure(self):
        if not self.ready or self.process is None or self.process.poll() is not None:
            self.start()

    def stop(self):
        self.ready = False
        if self.process is not None:
            try:
                if self.process.stdin: self.process.stdin.close()
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                try: self.process.wait(timeout=2)
                except subprocess.TimeoutExpired: pass
        self.process = None; self.pid = None

    def cancel(self):
        self.ready = False
        process = self.process
        self.process = None
        self.pid = None
        if process is not None:
            process.kill()
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired: pass

    def stream(self, prompt_ids, steps, chat, handler):
        request_id = self.next_id; self.next_id += 1
        self.process.stdin.write(f"{request_id} {steps} {','.join(map(str, prompt_ids))}\n")
        self.process.stdin.flush()
        generated, previous = [], ""
        latest_prefill = latest_summary = done_event = None
        seen_start = False; client_alive = True; started = time.perf_counter()
        while True:
            raw = self.process.stdout.readline()
            if not raw:
                self.ready = False; raise ConnectionError("persistent worker closed")
            line = raw.strip()
            if not line.startswith("{"):
                if line: print(f"device: {line[:240]}", flush=True)
                continue
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            kind = event.get("tensorfold")
            if kind == "qualcomm-exl3-worker-start":
                if int(event.get("id", -1)) != request_id: raise RuntimeError("worker request id mismatch")
                seen_start = True
            elif kind == "qualcomm-exl3-prefill": latest_prefill = event
            elif kind == "qualcomm-exl3-speculative-chunk":
                count = int(event["count"]); generated.extend(int(x) for x in event["tokens"][:count])
                full = decode_tokens(generated)
                piece = full[len(previous):] if full.startswith(previous) else full
                previous = full
                if not piece: piece = "\u200b"
                choice = ({"index": 0, "delta": {"content": piece}, "finish_reason": None}
                          if chat else {"index": 0, "text": piece, "finish_reason": None})
                if client_alive:
                    try: handler.send_event({"id": "tf-phone", "object": "chat.completion.chunk" if chat else "text_completion",
                                             "model": MODEL_ID, "choices": [choice]})
                    except (BrokenPipeError, ConnectionResetError):
                        self.cancel()
                        raise ConnectionAbortedError("client disconnected; worker cancelled")
            elif kind == "qualcomm-exl3-speculative-summary": latest_summary = event
            elif kind == "qualcomm-exl3-worker-done":
                if int(event.get("id", -1)) != request_id: raise RuntimeError("worker completion id mismatch")
                done_event = event; break
        if not seen_start or latest_summary is None or len(generated) != steps:
            raise RuntimeError(f"incomplete worker request {len(generated)}/{steps}")
        if client_alive:
            finish = ({"index": 0, "delta": {}, "finish_reason": "length"}
                      if chat else {"index": 0, "text": "", "finish_reason": "length"})
            prefill_s = (latest_prefill or {}).get("elapsed_ms", 0) / 1000
            decode_s = latest_summary.get("speculative_elapsed_ms", 0) / 1000
            handler.send_event({"id": "tf-phone", "object": "chat.completion.chunk" if chat else "text_completion",
                                "model": MODEL_ID, "choices": [finish],
                                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(generated),
                                          "total_tokens": len(prompt_ids) + len(generated)},
                                "runtime": {"engine": "tensorfold-qualcomm-exl3-local", "sampling": "greedy",
                                            "drafts": True, "prefill_seconds": prefill_s,
                                            "prompt_cache_hit": bool((latest_prefill or {}).get("cache_hit", False)),
                                            "prompt_cache_prefix_tokens": int((latest_prefill or {}).get("cache_prefix_tokens", 0)),
                                            "prefill_batch_tokens": int((latest_prefill or {}).get("batch", 0)),
                                            "decode_seconds": decode_s,
                                            "tokens_per_second": len(generated) / decode_s if decode_s > 0 else 0,
                                            "token_sha": token_sha(generated), "mtp_depth": MTP_DEPTH,
                                            "mtp_initial_rank": MTP_INITIAL_RANK,
                                            "rounds": latest_summary.get("rounds"),
                                            "accepted_drafts": latest_summary.get("accepted_drafts"),
                                            "serial_fallbacks": latest_summary.get("serial_fallbacks", 0),
                                            "resident_weight_uploads": done_event.get("resident_weight_uploads"),
                                            "mtp_projection_uploads": done_event.get("mtp_projection_uploads")}})
            handler.wfile.write(b"data: [DONE]\n\n"); handler.wfile.flush()
        print(f"request={request_id} worker={self.pid} prompt={len(prompt_ids)} output={len(generated)} "
              f"wall={time.perf_counter()-started:.3f}s", flush=True)


WORKER = PersistentWorker()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "tensorfold-qualcomm-local/1"
    def log_message(self, fmt, *args): print("http " + fmt % args, flush=True)
    def json_response(self, status, value):
        raw = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw))); self.send_header("Connection", "close")
        self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path == "/health":
            return self.json_response(200 if WORKER.ready else 503,
                                      {"status": "ok" if WORKER.ready else "starting", "worker_pid": WORKER.pid})
        if self.path == "/v1/models":
            return self.json_response(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        return self.json_response(404, {"error": {"message": "not found"}})
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            if self.path in ("/tokenize", "/v1/tokenize"):
                if body.get("messages") is not None:
                    options = body.get("chat_template_kwargs") or {}
                    text = TOKENIZER.chat_template(body["messages"], bool(options.get("enable_thinking", False)))
                else:
                    text = body.get("content", body.get("prompt", ""))
                return self.json_response(200, {"tokens": TOKENIZER.encode(text, bool(body.get("parse_special", True)))})
            if self.path in ("/detokenize", "/v1/detokenize"):
                return self.json_response(200, {"content": TOKENIZER.decode(body.get("tokens") or [])})
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
                return self.json_response(404, {"error": {"message": "not found"}})
            if float(body.get("temperature", 0)) != 0: return self.json_response(400, {"error": {"message": "greedy-only"}})
            if not body.get("stream"): return self.json_response(400, {"error": {"message": "stream=true required"}})
            steps = int(body.get("max_tokens", 256)); ids = encode_request(self.path, body)
            if steps < 1 or steps > MAX_STEPS or not ids or len(ids) > 128: raise ValueError("request outside worker capacity")
        except Exception as error: return self.json_response(400, {"error": {"message": str(error)}})
        with REQUEST_LOCK:
            WORKER.ensure(); self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
            try: WORKER.stream(ids, steps, self.path.endswith("/chat/completions"), self)
            except Exception as error:
                WORKER.ready = False
                try: self.send_event({"error": {"message": str(error)}}); self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError): pass
    def send_event(self, value):
        self.wfile.write(b"data: " + json.dumps(value, separators=(",", ":")).encode() + b"\n\n"); self.wfile.flush()


if __name__ == "__main__":
    try:
        WORKER.start(); print(f"TensorFold local API on {HTTP_HOST}:{HTTP_PORT}", flush=True)
        ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), Handler).serve_forever()
    finally:
        WORKER.stop()
