#!/usr/bin/env python3
"""OpenAI streaming adapter backed by one persistent Qualcomm TensorFold worker."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from transformers import AutoTokenizer
import hashlib, json, os, shlex, socket, subprocess, threading, time, sys

ADB = os.environ.get("TF_ADB", r"D:/projects/phoneproject/.tools/android-sdk/platform-tools/adb.exe")
SERIAL = os.environ.get("TF_SERIAL", "e1db9f1d")
MODEL_ID = os.environ.get("TF_MODEL_ID", "qwen35-4b-exl3-4bpw-persistent")
HTTP_PORT = int(os.environ.get("TF_HTTP_PORT", "18004"))
DEVICE_PORT = int(os.environ.get("TF_DEVICE_PORT", "19050"))
BUNDLE = os.environ.get("TF_BUNDLE", "/data/local/tmp/exl3-full-power")
BINARY = os.environ.get("TF_BINARY", BUNDLE + "/tensorfold-qualcomm-exl3-forward-4b-persistent-http-v1")
MODEL = os.environ.get("TF_MODEL", "/data/local/tmp/tensorfold-models/Qwen3.5-4B-exl3-4.00bpw/model.safetensors")
CACHE = os.environ.get("TF_CACHE", "/data/local/tmp/tensorfold-exl3-cache-4b-v1")
MTP = os.environ.get("TF_MTP", "/data/local/tmp/tensorfold-mtp-poc/mtp.safetensors")
MODEL_SHA = os.environ.get("TF_MODEL_SHA", "117eec31ed767c3acc4a6b56c37336f2bb56dd6d68daf058cec3d7f108e032d1")
MODE = os.environ.get("TF_MODE", "serve-q4-full-out-q8-q0-head-adaptive-linear")
MAX_STEPS = int(os.environ.get("TF_MAX_STEPS", "1024"))
PROFILE_PREFILL = os.environ.get("TF_PERSISTENT_PROFILE") == "1"
MTP_DEPTH = int(os.environ.get("TF_MTP_DEPTH", "2"))
MTP_INITIAL_RANK = int(os.environ.get("TF_MTP_INITIAL_RANK", "1"))
MTP_FIRST_MARGIN = os.environ.get("TF_MTP_FIRST_MARGIN", "0.1").strip() or None
MTP_SECOND_MARGIN = os.environ.get("TF_MTP_SECOND_MARGIN", "").strip() or None
TARGET_AMBIGUITY_MARGIN = os.environ.get("TF_TARGET_AMBIGUITY_MARGIN", "0.125").strip() or None
Q8_OUTPUT_END = int(os.environ.get("TF_Q8_OUTPUT_END", "24"))
FUSED_LINEAR_MASK = os.environ.get("TF_FUSED_LINEAR_MASK", "0")
HMX_MIN_ROWS = int(os.environ.get("TF_HMX_MIN_ROWS", "4"))
OPPOLL = int(os.environ.get("TF_OPPOLL", "1"))
POWER_MODE = os.environ.get("TF_POWER_MODE", "burst")
TOKENIZER = AutoTokenizer.from_pretrained(os.environ.get("TF_TOKENIZER", "Qwen/Qwen3.5-4B"))
REQUEST_LOCK = threading.Lock()


def token_sha(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def encode_request(path, body):
    if path.endswith("/chat/completions"):
        kw = body.get("chat_template_kwargs") or {}
        encoded = TOKENIZER.apply_chat_template(
            body.get("messages") or [], tokenize=True, add_generation_prompt=True,
            enable_thinking=bool(kw.get("enable_thinking", False)), return_dict=False)
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        return [int(x) for x in encoded]
    return [int(x) for x in TOKENIZER.encode(body.get("prompt", ""), add_special_tokens=True)]


class PersistentWorker:
    def __init__(self):
        self.launcher = None
        self.sock = None
        self.source = None
        self.sink = None
        self.pid = None
        self.ready = False
        self.next_id = 1

    def command(self):
        margins = (
            (f"TENSORFOLD_MTP_FIRST_MARGIN={MTP_FIRST_MARGIN} " if MTP_FIRST_MARGIN is not None else "") +
            (f"TENSORFOLD_MTP_SECOND_MARGIN={MTP_SECOND_MARGIN} " if MTP_SECOND_MARGIN is not None else "") +
            (f"TENSORFOLD_TARGET_AMBIGUITY_MARGIN={TARGET_AMBIGUITY_MARGIN} " if TARGET_AMBIGUITY_MARGIN is not None else "")
        )
        env = (
            f"cd {BUNDLE}; "
            f"export LD_LIBRARY_PATH={BUNDLE}/lib ADSP_LIBRARY_PATH={BUNDLE}/lib "
            f"GGML_HEXAGON_OPPOLL={OPPOLL} GGML_HEXAGON_HMX_PIPELINE_MIN_ROWS={HMX_MIN_ROWS} "
            f"GGML_HEXAGON_POWER_MODE={POWER_MODE} "
            f"TENSORFOLD_MTP_SIDECAR={MTP} "
            f"TENSORFOLD_MTP_DEPTH={MTP_DEPTH} TENSORFOLD_MTP_INITIAL_RANK={MTP_INITIAL_RANK} "
            f"TENSORFOLD_Q8_OUTPUT_LAYER_END={Q8_OUTPUT_END} TENSORFOLD_FUSED_LINEAR_LAYER_MASK={FUSED_LINEAR_MASK} "
            f"{margins}"
            "TENSORFOLD_BENCH_QUIET=1 TENSORFOLD_STREAM_TOKENS=1 "
            f"{'TENSORFOLD_PREFILL_PROFILE=1 ' if PROFILE_PREFILL else ''}"
            "TENSORFOLD_PERSISTENT_WORKER=1; "
        )
        argv = (
            f"{BINARY} htp {MODEL} 0 6 {CACHE} {MODEL_SHA} {MAX_STEPS} 0 {MODE} 1"
        )
        return f"printf 'WORKER_PID %d\\n' $$; {env} exec {argv}"

    def start(self):
        self.stop()
        local = f"tcp:{DEVICE_PORT}"
        subprocess.run([ADB, "-s", SERIAL, "forward", "--remove", local],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([ADB, "-s", SERIAL, "forward", local, local], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        listener = (
            f"exec toybox nc -s 127.0.0.1 -p {DEVICE_PORT} -l -E sh -c "
            f"{shlex.quote(self.command())}"
        )
        self.launcher = subprocess.Popen([ADB, "-s", SERIAL, "shell", listener],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.4)
        for _ in range(60):
            try:
                self.sock = socket.create_connection(("127.0.0.1", DEVICE_PORT), timeout=0.5)
                break
            except OSError:
                time.sleep(0.1)
        if self.sock is None:
            raise ConnectionError("persistent worker listener did not start")
        self.sock.settimeout(None)
        self.source = self.sock.makefile("r", encoding="utf-8", errors="replace", newline="\n")
        self.sink = self.sock.makefile("w", encoding="utf-8", newline="\n")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            line = self.source.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("WORKER_PID "):
                self.pid = int(line.split()[1])
                continue
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
        if not self.ready or self.launcher is None or self.launcher.poll() is not None:
            self.start()

    def stop(self):
        self.ready = False
        for stream in (self.sink, self.source):
            if stream is not None:
                try: stream.close()
                except OSError: pass
        self.sink = self.source = None
        if self.sock is not None:
            try: self.sock.close()
            except OSError: pass
        self.sock = None
        if self.launcher is not None:
            try: self.launcher.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.launcher.kill()
        self.launcher = None
        if self.pid is not None:
            subprocess.run([ADB, "-s", SERIAL, "shell", f"kill {self.pid}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        self.pid = None
        subprocess.run([ADB, "-s", SERIAL, "forward", "--remove", f"tcp:{DEVICE_PORT}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stream(self, prompt_ids, steps, chat, handler):
        request_id = self.next_id
        self.next_id += 1
        self.sink.write(f"{request_id} {steps} {','.join(map(str, prompt_ids))}\n")
        self.sink.flush()
        generated, previous = [], ""
        latest_prefill = latest_summary = done_event = None
        seen_start = False
        client_alive = True
        started = time.perf_counter()
        while True:
            raw = self.source.readline()
            if not raw:
                self.ready = False
                raise ConnectionError("persistent worker closed")
            line = raw.strip()
            if not line.startswith("{"):
                if line: print(f"device: {line[:240]}", flush=True)
                continue
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            kind = event.get("tensorfold")
            if kind == "qualcomm-exl3-worker-start":
                if int(event.get("id", -1)) != request_id:
                    raise RuntimeError("worker request id mismatch")
                seen_start = True
            elif kind == "qualcomm-exl3-prefill":
                latest_prefill = event
            elif kind == "qualcomm-exl3-speculative-chunk":
                count = int(event["count"])
                for encoded in event["tokens"][:count]:
                    tid = int(encoded); generated.append(tid)
                    full = TOKENIZER.decode(generated, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
                    piece = full[len(previous):] if full.startswith(previous) else full
                    previous = full
                    if not piece: piece = "\u200b"
                    choice = ({"index": 0, "delta": {"content": piece}, "finish_reason": None}
                              if chat else {"index": 0, "text": piece, "finish_reason": None})
                    if client_alive:
                        try:
                            handler.send_event({"id": "tf-phone", "object": "chat.completion.chunk" if chat else "text_completion",
                                                "model": MODEL_ID, "choices": [choice]})
                        except (BrokenPipeError, ConnectionResetError):
                            self.stop()
                            raise ConnectionAbortedError("client disconnected; worker cancelled")
            elif kind == "qualcomm-exl3-speculative-summary":
                latest_summary = event
            elif kind == "qualcomm-exl3-worker-done":
                if int(event.get("id", -1)) != request_id:
                    raise RuntimeError("worker completion id mismatch")
                done_event = event
                break
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
                                "runtime": {"engine": "tensorfold-qualcomm-exl3-persistent", "sampling": "greedy",
                                            "drafts": True, "prefill_seconds": prefill_s,
                                            "prefill_projection_wall_seconds": (latest_prefill or {}).get("projection_wall_ms", 0) / 1000,
                                            "prefill_compute_seconds": (latest_prefill or {}).get("matmul_compute_ms", 0) / 1000,
                                            "prefill_head_seconds": (latest_prefill or {}).get("head_ms", 0) / 1000,
                                            "decode_seconds": decode_s,
                                            "tokens_per_second": len(generated) / decode_s if decode_s > 0 else 0,
                                            "token_sha": token_sha(generated),
                                            "mtp_depth": MTP_DEPTH, "mtp_initial_rank": MTP_INITIAL_RANK,
                                            "rounds": latest_summary.get("rounds"), "accepted_drafts": latest_summary.get("accepted_drafts"),
                                            "adaptive_depth1_rounds": latest_summary.get("adaptive_depth1_rounds"),
                                            "adaptive_depth2_rounds": latest_summary.get("adaptive_depth2_rounds"),
                                            "serial_fallbacks": latest_summary.get("serial_fallbacks", 0),
                                            "mtp_compute_seconds": latest_summary.get("mtp_compute_ms", 0) / 1000,
                                            "request_seconds": latest_summary.get("request_elapsed_ms", 0) / 1000,
                                            "resident_weight_uploads": done_event.get("resident_weight_uploads"),
                                            "resident_weight_hits": done_event.get("resident_weight_hits"),
                                            "mtp_projection_uploads": done_event.get("mtp_projection_uploads")}})
            handler.wfile.write(b"data: [DONE]\n\n"); handler.wfile.flush()
        print(f"request={request_id} pid={self.pid} prompt={len(prompt_ids)} output={len(generated)} wall={time.perf_counter()-started:.3f}s uploads={done_event.get('resident_weight_uploads')} hits={done_event.get('resident_weight_hits')} mtp_uploads={done_event.get('mtp_projection_uploads')}", flush=True)


WORKER = PersistentWorker()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "tensorfold-qualcomm-persistent/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("persistent-shim " + fmt % args + "\n")

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
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            return self.json_response(404, {"error": {"message": "not found"}})
        try:
            n = int(self.headers.get("Content-Length", "0")); body = json.loads(self.rfile.read(n))
            if float(body.get("temperature", 0)) != 0:
                return self.json_response(400, {"error": {"message": "Qualcomm TensorFold port is greedy-only"}})
            if not body.get("stream"):
                return self.json_response(400, {"error": {"message": "stream=true is required"}})
            steps = int(body.get("max_tokens", 256))
            if steps < 1 or steps > MAX_STEPS: raise ValueError("max_tokens outside worker capacity")
            ids = encode_request(self.path, body)
            if not ids or len(ids) > 128: raise ValueError("prompt must contain 1..128 tokens")
        except Exception as e:
            return self.json_response(400, {"error": {"message": str(e)}})
        with REQUEST_LOCK:
            WORKER.ensure()
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close")
            self.end_headers()
            try:
                WORKER.stream(ids, steps, self.path.endswith("/chat/completions"), self)
            except Exception as e:
                WORKER.ready = False
                try:
                    self.send_event({"error": {"message": str(e)}})
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError): pass

    def send_event(self, value):
        self.wfile.write(b"data: " + json.dumps(value, separators=(",", ":")).encode() + b"\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    try:
        WORKER.start()
        print(f"Persistent TF adapter on 0.0.0.0:{HTTP_PORT}", flush=True)
        ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler).serve_forever()
    finally:
        WORKER.stop()
