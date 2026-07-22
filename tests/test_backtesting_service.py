import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtesting_service import build_signal_frame, run_backtest, simulate_backtest


def synthetic_history(rows: int = 300) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=rows)
    trend = np.linspace(100, 170, rows)
    cycle = np.sin(np.arange(rows) / 8) * 4
    return pd.DataFrame(
        {"Close": trend + cycle, "Volume": np.full(rows, 1_000_000)},
        index=index,
    )


class BacktestingServiceTests(unittest.TestCase):
    def test_position_is_previous_days_signal(self):
        frame = build_signal_frame(synthetic_history(), transaction_cost_bps=0)
        expected = frame["signal"].shift(1)
        comparable = expected.notna()
        self.assertTrue((frame.loc[comparable, "position"] == expected[comparable]).all())

    def test_transaction_costs_reduce_ending_value(self):
        history = synthetic_history()
        free = simulate_backtest(history, transaction_cost_bps=0)
        costly = simulate_backtest(history, transaction_cost_bps=25)
        self.assertLessEqual(costly["strategy"]["ending_value"], free["strategy"]["ending_value"])

    def test_response_contains_benchmark_and_bounded_curve(self):
        result = simulate_backtest(synthetic_history(800))
        self.assertIn("strategy", result)
        self.assertIn("benchmark", result)
        self.assertLessEqual(len(result["equity_curve"]), 501)
        self.assertGreater(result["trading_days"], 0)
        self.assertEqual(result["equity_curve"][0]["strategy"], 10_000)
        self.assertEqual(result["equity_curve"][0]["benchmark"], 10_000)
        expected_return = (result["strategy"]["ending_value"] / 10_000 - 1) * 100
        self.assertAlmostEqual(result["strategy"]["total_return_percent"], expected_return, places=1)

    def test_run_backtest_exposes_auditable_proof(self):
        with patch("backtesting_service.yf.Ticker") as ticker:
            ticker.return_value.history.return_value = synthetic_history()
            result = run_backtest("AAPL", period="2y", initial_capital=5000)

        self.assertEqual(result["proof"]["input_parameters"]["initial_capital"], 5000)
        self.assertTrue(result["proof"]["adjusted_prices"])
        self.assertIn("look_ahead_prevention", result["proof"]["integrity_checks"])
        expected = round(
            result["overview"]["strategy_return_percent"]
            - result["overview"]["benchmark_return_percent"],
            2,
        )
        self.assertEqual(result["overview"]["difference_percentage_points"], expected)


if __name__ == "__main__":
    unittest.main()
