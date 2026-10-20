#!/usr/bin/env python3
"""Pure-stdlib Qwen3.5 ByteLevel-BPE tokenizer used by the Qualcomm phone server."""
from functools import lru_cache
import json, pathlib, unicodedata

CONTRACTIONS = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]; n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {char: byte for byte, char in BYTE_ENCODER.items()}


def _kind(char):
    return unicodedata.category(char)[0]


def _is_letter_mark(char):
    return _kind(char) in ("L", "M")


def _is_number(char):
    return _kind(char) == "N"


def _pieces(text):
    """The tokenizer.json Split regex, behavior=Isolated, without a third-party regex module."""
    out = []; i = 0; length = len(text)
    while i < length:
        lower = text[i:i + 3].lower()
        contraction = next((value for value in CONTRACTIONS if lower.startswith(value)), None)
        if contraction is not None:
            out.append(text[i:i + len(contraction)]); i += len(contraction); continue

        start = i
        if _is_letter_mark(text[i]):
            i += 1
            while i < length and _is_letter_mark(text[i]): i += 1
            out.append(text[start:i]); continue
        if text[i] not in "\r\n" and not _is_number(text[i]) and _kind(text[i]) != "L" and i + 1 < length and _is_letter_mark(text[i + 1]):
            i += 2
            while i < length and _is_letter_mark(text[i]): i += 1
            out.append(text[start:i]); continue

        if _is_number(text[i]):
            out.append(text[i]); i += 1; continue

        start = i
        if text[i] == " " and i + 1 < length and not text[i + 1].isspace() and not _is_letter_mark(text[i + 1]) and not _is_number(text[i + 1]):
            i += 1
        punct = i
        while i < length and not text[i].isspace() and not _is_letter_mark(text[i]) and not _is_number(text[i]): i += 1
        if i > punct:
            while i < length and text[i] in "\r\n": i += 1
            out.append(text[start:i]); continue
        i = start

        if text[i].isspace():
            end = i
            while end < length and text[end].isspace(): end += 1
            last_newline = max(text.rfind("\r", i, end), text.rfind("\n", i, end))
            if last_newline >= i:
                out.append(text[i:last_newline + 1]); i = last_newline + 1; continue
            if end == length:
                out.append(text[i:end]); i = end; continue
            if end - i > 1:
                out.append(text[i:end - 1]); i = end - 1; continue
            out.append(text[i:end]); i = end; continue

        # The JSON regex is exhaustive; reaching this means a new Unicode category needs classification.
        out.append(text[i]); i += 1
    return out


class Qwen35Tokenizer:
    def __init__(self, directory):
        directory = pathlib.Path(directory)
        self.vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        self.ids = {value: key for key, value in self.vocab.items()}
        merges = []
        for line in (directory / "merges.txt").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"): continue
            left, right = line.split(" ", 1); merges.append((left, right))
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        spec = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        self.added = {item["content"]: int(item["id"]) for item in spec.get("added_tokens", [])}
        self.added_ids = {value: key for key, value in self.added.items()}
        self._added_order = sorted(self.added, key=len, reverse=True)

    @lru_cache(maxsize=65536)
    def _bpe(self, word):
        symbols = tuple(word)
        if len(symbols) < 2: return symbols
        while True:
            pairs = {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}
            best = min(pairs, key=lambda pair: self.ranks.get(pair, 1 << 60))
            if best not in self.ranks: break
            merged = []; i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == best[0] and symbols[i + 1] == best[1]:
                    merged.append(best[0] + best[1]); i += 2
                else:
                    merged.append(symbols[i]); i += 1
            symbols = tuple(merged)
            if len(symbols) < 2: break
        return symbols

    def _ordinary(self, text):
        result = []
        for piece in _pieces(unicodedata.normalize("NFC", text)):
            encoded = "".join(BYTE_ENCODER[byte] for byte in piece.encode("utf-8"))
            for token in self._bpe(encoded):
                result.append(self.vocab[token])
        return result

    def encode(self, text, parse_special=True):
        if not parse_special: return self._ordinary(text)
        result = []; ordinary = []; i = 0
        while i < len(text):
            token = next((value for value in self._added_order if text.startswith(value, i)), None)
            if token is None:
                ordinary.append(text[i]); i += 1; continue
            if ordinary:
                result.extend(self._ordinary("".join(ordinary))); ordinary.clear()
            result.append(self.added[token]); i += len(token)
        if ordinary: result.extend(self._ordinary("".join(ordinary)))
        return result

    def decode(self, token_ids):
        out = []; raw = bytearray()
        def flush():
            if raw:
                out.append(raw.decode("utf-8", errors="replace")); raw.clear()
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self.added_ids:
                flush(); out.append(self.added_ids[token_id]); continue
            token = self.ids[token_id]
            for char in token: raw.append(BYTE_DECODER[char])
        flush()
        return "".join(out)

    @staticmethod
    def chat_template(messages, enable_thinking=False):
        parts = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role not in ("system", "user", "assistant", "tool") or not isinstance(content, str):
                raise ValueError("local tokenizer supports text system/user/assistant/tool messages")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
        return "".join(parts)
