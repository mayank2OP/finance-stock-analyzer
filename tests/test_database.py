import unittest

from database import normalize_database_url


class DatabaseConfigurationTests(unittest.TestCase):
    def test_normalizes_neon_postgresql_url_to_psycopg3(self):
        url = "postgresql://user:password@example.neon.tech/stockpilot?sslmode=require"
        self.assertEqual(
            normalize_database_url(url),
            "postgresql+psycopg://user:password@example.neon.tech/stockpilot?sslmode=require",
        )

    def test_keeps_sqlite_url_unchanged(self):
        self.assertEqual(
            normalize_database_url("sqlite:///./stock_analyser.db"),
            "sqlite:///./stock_analyser.db",
        )


if __name__ == "__main__":
    unittest.main()
