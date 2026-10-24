#!/usr/bin/env python3
"""Pure stdlib checks for the deterministic QNN full-block fixture generator."""
from __future__ import annotations

import hashlib
import unittest

import generate_qnn_full_projection as fixture


class FullProjectionFixtureTest(unittest.TestCase):
    def test_checked_in_header_is_exact_regeneration(self) -> None:
        source = fixture.FIXTURE.read_bytes()
        self.assertEqual(fixture.FIXTURE_SHA256, hashlib.sha256(source).hexdigest())
        self.assertEqual(fixture.render(source), fixture.OUTPUT.read_text(encoding="utf-8"))

    def test_full_block_has_bounded_quantization(self) -> None:
        _, values = fixture.oracle_block(fixture.FIXTURE.read_bytes())
        scale = max(abs(value) for value in values) / 127.0
        quantized = fixture.quantize_symmetric(values, scale)
        self.assertEqual(128 * 128, len(quantized))
        self.assertTrue(all(1 <= value <= 255 for value in quantized))


if __name__ == "__main__":
    unittest.main()
