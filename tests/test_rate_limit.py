import unittest

from fastapi import HTTPException

from rate_limit import clear_rate_limits, enforce_rate_limit


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        clear_rate_limits()

    def test_rejects_request_after_limit(self):
        enforce_rate_limit("test:key", limit=1, window_seconds=60)
        with self.assertRaises(HTTPException) as raised:
            enforce_rate_limit("test:key", limit=1, window_seconds=60)
        self.assertEqual(raised.exception.status_code, 429)
        self.assertIn("Retry-After", raised.exception.headers)


if __name__ == "__main__":
    unittest.main()
