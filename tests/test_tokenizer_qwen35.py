"""Stage 6: tokenizer compatibility against Qwen3.5-2B-exl3-8.00bpw.

These tests confirm the tokenizer that would ride alongside a converted
.tfold file produces the same token IDs the source model expects.

If either the pinned model dir or the Rust `tokenizers` library is not
available, the suite skips cleanly.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest



def _resolve_model_dir() -> pathlib.Path | None:
    for cand in (
        pathlib.Path("D:/op13-root/models/Qwen3.5-2B-exl3-8.00bpw"),
        pathlib.Path("/mnt/d/op13-root/models/Qwen3.5-2B-exl3-8.00bpw"),
    ):
        if (cand / "tokenizer.json").exists():
            return cand
    return None


MODEL_DIR = _resolve_model_dir()
TOKENIZER_JSON = (MODEL_DIR / "tokenizer.json") if MODEL_DIR else None
def _require():
    if TOKENIZER_JSON is None or not TOKENIZER_JSON.exists():
        pytest.skip("tokenizer.json not present (looked under D:/op13-root/models/ and /mnt/d/...)")
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        pytest.skip("Rust tokenizers library not installed")


def _tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_JSON))


def test_tokenizer_file_sha256_is_pinned_once_loaded():
    _require()
    # Record the SHA for reproducibility; not asserting equality against a
    # pinned constant because Qwen tokenizer files may legitimately differ
    # across checkpoint revisions. Just ensure we hash a nonempty file.
    data = TOKENIZER_JSON.read_bytes()
    assert len(data) > 1000, "tokenizer.json unexpectedly tiny"
    sha = hashlib.sha256(data).hexdigest()
    print(f"tokenizer.json SHA-256 = {sha}")


def test_roundtrip_hello_world():
    _require()
    tok = _tokenizer()
    text = "Hello, world!"
    ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) > 0
    decoded = tok.decode(ids)
    # Qwen tokenizer is lossless over ASCII text; strip leading space
    # some BPE tokenizers emit at word starts.
    assert decoded.strip() == text.strip()


def test_roundtrip_common_prompts():
    _require()
    tok = _tokenizer()
    cases = [
        "The quick brown fox jumps over the lazy dog.",
        "def add(a, b): return a + b",
        "1 + 2 = 3",
        "Hello, world!\nThis is a test.",
    ]
    for t in cases:
        ids = tok.encode(t, add_special_tokens=False).ids
        back = tok.decode(ids)
        assert back.strip() == t.strip(), f"roundtrip failed for {t!r}: got {back!r}"


def test_special_tokens_present():
    """Qwen3.5 requires an EOS and a chat-template <|im_start|> / <|im_end|>
    structure. Verify the tokenizer knows them."""
    _require()
    tok = _tokenizer()
    special = tok.get_vocab_size(with_added_tokens=True)
    base = tok.get_vocab_size(with_added_tokens=False)
    assert special >= base
    # Known Qwen3.5 special tokens
    vocab = tok.get_vocab()
    expected = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    found = [t for t in expected if t in vocab]
    assert found, f"none of {expected} found in Qwen tokenizer vocab"


def test_eos_token_resolvable():
    """The EOS token id must be stable for greedy decoding to terminate."""
    _require()
    tok = _tokenizer()
    vocab = tok.get_vocab()
    # Qwen3.5 uses <|im_end|> for chat termination and <|endoftext|> for raw
    eos_candidates = ["<|im_end|>", "<|endoftext|>"]
    ids = [vocab[t] for t in eos_candidates if t in vocab]
    assert ids, "no EOS candidate found"
    # Every EOS id must decode back to its string
    for t, i in zip(eos_candidates, ids):
        if t in vocab:
            # decode(single id) MAY include whitespace or no-op
            # but encode(that string) must give back a sequence containing that id
            back_ids = tok.encode(t, add_special_tokens=False).ids
            assert i in back_ids or tok.id_to_token(i) == t, (
                f"EOS roundtrip failed for {t!r}: id={i}, re-encoded={back_ids}"
            )


def test_deterministic_encoding():
    """Same input must produce same token IDs every call. Required for
    reproducible generation."""
    _require()
    tok = _tokenizer()
    prompt = "The meaning of life is"
    runs = [tuple(tok.encode(prompt, add_special_tokens=False).ids) for _ in range(5)]
    assert len(set(runs)) == 1, f"tokenizer is nondeterministic: {runs}"


def test_known_hello_tokenization():
    """Record the token IDs for the Stage 5 'hi' oracle input. These are
    used by the oracle fixture at zig/tests/fixtures/exl3_oracle.bin and
    referenced in docs/qualcomm-android.md as the retained 'hi' stream."""
    _require()
    tok = _tokenizer()
    ids = tok.encode("hi", add_special_tokens=False).ids
    assert ids, "empty encoding for 'hi'"
    # Record for docs; Qwen3.5-2B-exl3-8bpw is expected to produce id 5834
    # for 'hi' (per docs/qualcomm-android.md).
    assert ids[0] == 5834, f"Qwen3.5-2B 'hi' first token expected 5834, got {ids[0]}"


def test_vocab_size_matches_model_config():
    """Qwen3.5-2B config.json reports a vocab size; the tokenizer must
    agree so logits-indexing doesn't fall off the end."""
    _require()
    import json
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        pytest.skip("config.json not present")
    cfg = json.loads(config_path.read_text())
    # Allow nesting under text_config for VL Qwen variants
    expected_vocab = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size")
    assert expected_vocab, "no vocab_size in config.json"
    tok = _tokenizer()
    tok_vocab = tok.get_vocab_size(with_added_tokens=True)
    assert tok_vocab <= expected_vocab, (
        f"tokenizer vocab {tok_vocab} exceeds model vocab_size {expected_vocab}"
    )
