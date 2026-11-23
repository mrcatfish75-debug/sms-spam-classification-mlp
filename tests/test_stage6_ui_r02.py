"""Synthetic-only UI regression tests; never open train/validation/test datasets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402


class TestStage6UIR02(unittest.TestCase):
    def test_css_asset_present_and_safe(self):
        self.assertTrue(app.CSS_PATH.is_file())
        css = app.CSS_PATH.read_text(encoding="utf-8")
        self.assertIn(".score-card", css)
        self.assertIn("color: var(--sms-text) !important", css)
        self.assertIn(".score-threshold", css)

    def test_numeric_decisions_stay_at_frozen_half_threshold(self):
        self.assertEqual(app.THRESHOLD, 0.5)
        self.assertEqual(app.friendly_label(0.499999)[0], "Pesan normal (Ham)")
        self.assertEqual(app.friendly_label(0.5)[0], "Terindikasi spam")
        self.assertEqual(app.percent(0.0005), "0,05%")

    def test_displayed_scale_marker_and_precise_percent(self):
        card = app.score_card("GRU + Attention (UAS)", 0.0005)
        self.assertIn("0,05%", card)
        self.assertIn("Ambang 50%", card)
        self.assertIn('style="width:0.050000%"', card)
        self.assertNotIn("keyakinan", card.lower())

    def test_attention_is_escaped_and_weights_are_original(self):
        markup = app.attention_view({"top_attention": [
            {"token": "<img src=x onerror=alert(1)>", "weight": 0.2},
            {"token": "today", "weight": 0.1},
        ]}, True)
        self.assertNotIn("<img src=x", markup)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", markup)
        self.assertIn("0.200", markup)
        self.assertIn("0.100", markup)
        self.assertIn('style="width:50.000%"', markup)
        self.assertIn("bukan bukti", markup)

    def test_invalid_input_does_not_invoke_either_model(self):
        with patch.object(app, "predict_gru") as gru, patch.object(app, "predict_mlp") as mlp:
            summary, chart, details, attention = app.analyze("   ", app.MODEL_BOTH, True)
        self.assertIn("Pesan belum dapat dianalisis", summary)
        self.assertEqual((chart, details, attention), ("", "", ""))
        gru.assert_not_called()
        mlp.assert_not_called()

    def test_single_gru_view_no_attention_unless_requested(self):
        value = {
            "probability_spam": 0.0005,
            "token_count": 7,
            "unknown_count": 0,
            "unknown_fraction": 0.0,
            "truncated": False,
            "top_attention": [],
        }
        with patch.object(app, "predict_gru", return_value=value) as gru:
            with patch.object(app, "predict_mlp") as mlp:
                summary, chart, details, attention = app.analyze(
                    "Synthetic example", app.MODEL_GRU, False
                )
        self.assertIn("Pesan normal (Ham)", summary)
        self.assertIn("0,05%", chart)
        self.assertIn("Token dipakai", details)
        self.assertEqual(attention, "")
        gru.assert_called_once_with("Synthetic example", with_attention=False)
        mlp.assert_not_called()

    def test_comparison_keeps_two_independent_scorecards(self):
        gru_data = {
            "probability_spam": 0.2,
            "token_count": 3,
            "unknown_count": 1,
            "unknown_fraction": 1/3,
            "truncated": False,
            "top_attention": [],
        }
        with patch.object(app, "predict_gru", return_value=gru_data):
            with patch.object(app, "predict_mlp", return_value=0.8):
                summary, chart, details, attention = app.analyze(
                    "Synthetic example", app.MODEL_BOTH, False
                )
        self.assertIn("berbeda pendapat", summary)
        self.assertEqual(chart.count('class="score-card'), 2)
        self.assertIn("20,00%", chart)
        self.assertIn("80,00%", chart)
        self.assertIn("bukan perbandingan akurasi", details)
        self.assertEqual(attention, "")

    def test_mlp_attention_not_misrepresented(self):
        with patch.object(app, "predict_mlp", return_value=0.8):
            _, _, _, attention = app.analyze("Synthetic example", app.MODEL_MLP, True)
        self.assertIn("hanya tersedia", attention)


if __name__ == "__main__":
    unittest.main()
