import unittest

import numpy as np
import pandas as pd

from analysis_service import (
    _fallback_narrative,
    _clean_text,
    _parse_news_articles,
    _published_at_iso,
    _parse_json_object,
    calculate_confidence,
    calculate_metrics,
    score_evidence,
)


class AnalysisServiceTests(unittest.TestCase):
    def test_metrics_and_score_are_deterministic(self):
        prices = np.linspace(100, 140, 90)
        history = pd.DataFrame({
            "Close": prices,
            "Volume": np.full(90, 1_000_000),
        })

        metrics = calculate_metrics(history)
        first = score_evidence(metrics)
        second = score_evidence(metrics)

        self.assertEqual(metrics["trend"], "BULLISH")
        self.assertEqual(first, second)
        self.assertIn(first["action"], {"BUY", "HOLD", "SELL"})
        self.assertIn(first["risk"], {"LOW", "MEDIUM", "HIGH"})
        self.assertEqual(
            first["score"],
            sum(rule["score_contribution"] for rule in first["rule_trace"]),
        )
        self.assertIn("buy_score_min", first["decision_boundaries"])

    def test_rejects_insufficient_history(self):
        history = pd.DataFrame({"Close": [1, 2], "Volume": [10, 20]})
        with self.assertRaisesRegex(ValueError, "50 trading days"):
            calculate_metrics(history)

    def test_agent_json_parser_handles_wrapped_output(self):
        parsed = _parse_json_object('```json\n{"summary":"ok"}\n```')
        self.assertEqual(parsed, {"summary": "ok"})

    def test_fallback_exposes_agent_status(self):
        evidence = {
            "scoring": {"action": "HOLD", "risk": "MEDIUM", "reasons": []},
            "metrics": {"trend": "MIXED", "rsi_14": 50, "annualized_volatility_percent": 25},
        }
        fallback = _fallback_narrative(evidence)
        self.assertEqual(fallback["agent_analysis"]["status"], "fallback")

    def test_confidence_is_deterministic_and_explained(self):
        evidence = {
            "metrics": {
                "current_price": 100,
                "ma_20": 98,
                "ma_50": 95,
                "rsi_14": 55,
                "macd": 1.2,
                "macd_signal": 1.0,
            },
            "company": {"name": "Example Inc."},
            "news": [{"title": "One"}, {"title": "Two"}],
        }
        first = calculate_confidence(evidence, "completed")
        second = calculate_confidence(evidence, "completed")
        self.assertEqual(first, second)
        self.assertEqual(first[0], sum(first[1].values()))

    def test_source_text_and_timestamp_are_normalized(self):
        self.assertEqual(_clean_text("Appleâ€™s ecosystem"), "Apple’s ecosystem")
        self.assertEqual(_clean_text("volatility Ã— sqrt"), "volatility × sqrt")
        self.assertEqual(_published_at_iso(0), "1970-01-01T00:00:00+00:00")

    def test_news_parser_supports_flat_and_nested_yahoo_payloads(self):
        articles = [
            {
                "title": "Flat article",
                "publisher": "Publisher One",
                "link": "https://finance.yahoo.com/flat",
                "providerPublishTime": 0,
            },
            {
                "content": {
                    "title": "Nested article",
                    "provider": {"displayName": "Publisher Two"},
                    "pubDate": "2026-07-22T12:00:00Z",
                    "canonicalUrl": {"url": "https://finance.yahoo.com/nested"},
                }
            },
            {"title": "No attributable URL", "publisher": "Unknown"},
        ]

        parsed = _parse_news_articles(articles)

        self.assertEqual([item["title"] for item in parsed], ["Flat article", "Nested article"])
        self.assertEqual(parsed[0]["published_at"], "1970-01-01T00:00:00+00:00")
        self.assertEqual(parsed[1]["publisher"], "Publisher Two")


if __name__ == "__main__":
    unittest.main()
