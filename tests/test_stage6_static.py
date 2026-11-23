"""Offline tests for frozen Stage 6 preprocessing (no TensorFlow / dataset)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.uas_inference import (  # noqa: E402
    encode_sms,
    frozen_artifacts,
    label_and_confidence,
    validate_message,
)


class TestFrozenInference(unittest.TestCase):
    def test_sha256_and_freeze(self):
        artifacts = frozen_artifacts()
        self.assertEqual(len(artifacts.vocabulary), 3443)
        self.assertEqual(artifacts.sequence_length, 60)

    def test_unicode_casefold_and_post_padding(self):
        encoded = encode_sms("ＨＩ 😊")
        self.assertEqual(encoded.tokens, ("hi", "😊"))
        self.assertEqual(encoded.sequence.shape, (1, 60))
        self.assertTrue((encoded.sequence[0, 2:] == 0).all())

    def test_preserves_punctuation(self):
        encoded = encode_sms("HELLO!!!")
        self.assertEqual(encoded.tokens, ("hello", "!", "!", "!"))

    def test_truncation_is_post_60(self):
        encoded = encode_sms("a " * 65)
        self.assertEqual(encoded.tokens, ("a",) * 60)
        self.assertTrue(encoded.truncated)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            validate_message("   ")

    def test_too_long_rejected(self):
        with self.assertRaises(ValueError):
            validate_message("x" * 2001)

    def test_threshold_exactly_half(self):
        self.assertEqual(label_and_confidence(0.5)[0], "Spam")
        self.assertEqual(label_and_confidence(0.49999)[0], "Ham")


if __name__ == "__main__":
    unittest.main()
