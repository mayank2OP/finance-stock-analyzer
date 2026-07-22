import os
import unittest
from unittest.mock import patch

from stock_crew import app
from test_support import create_auth_headers, isolated_client


client = isolated_client(app)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = create_auth_headers(client)

    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy", "database": "healthy"})

    def test_analysis_requires_authentication(self):
        response = client.post("/analyze", json={"ticker": "AAPL"})
        self.assertEqual(response.status_code, 401)

    def test_normalizes_ticker_and_returns_analysis(self):
        payload = {"ticker": "AAPL", "action": "HOLD"}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "stock_crew.run_analysis", return_value=payload
        ) as mocked:
            response = client.post("/analyze", json={"ticker": " aapl "}, headers=self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "data": payload})
        mocked.assert_called_once_with("AAPL")

    def test_rejects_invalid_ticker(self):
        response = client.post("/analyze", json={"ticker": "AAPL; DROP"}, headers=self.headers)
        self.assertEqual(response.status_code, 422)

    def test_timeout_returns_gateway_timeout(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "stock_crew.run_analysis", side_effect=TimeoutError("market data timed out")
        ):
            response = client.post("/analyze", json={"ticker": "AAPL"}, headers=self.headers)

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["detail"], "market data timed out")

    def test_backtest_contract(self):
        payload = {"ticker": "AAPL", "strategy": {"total_return_percent": 12.5}}
        with patch("stock_crew.run_backtest", return_value=payload) as mocked:
            response = client.post(
                "/backtest",
                json={
                    "ticker": "aapl",
                    "period": "5y",
                    "initial_capital": 10000,
                    "transaction_cost_bps": 5,
                    "horizon_days": 20,
                },
                headers=self.headers,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "data": payload})
        mocked.assert_called_once_with(
            ticker="AAPL",
            period="5y",
            initial_capital=10000,
            transaction_cost_bps=5,
            horizon_days=20,
            include_equity_curve=False,
        )

    def test_backtest_rejects_unsupported_period(self):
        response = client.post(
            "/backtest", json={"ticker": "AAPL", "period": "50y"}, headers=self.headers
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
