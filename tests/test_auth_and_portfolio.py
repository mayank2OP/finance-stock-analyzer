import unittest
import uuid

from stock_crew import app
from test_support import isolated_client


client = isolated_client(app)


class AuthAndPortfolioTests(unittest.TestCase):
    def _register_and_login(self) -> tuple[str, str]:
        username = f"test_{uuid.uuid4().hex[:12]}"
        password = "correct-horse-42"
        registered = client.post("/auth/register", json={"username": username, "password": password})
        self.assertEqual(registered.status_code, 201)
        self.assertNotIn("password", str(registered.json()).lower())

        logged_in = client.post(
            "/auth/token",
            data={"username": username, "password": password},
        )
        self.assertEqual(logged_in.status_code, 200)
        return username, logged_in.json()["access_token"]

    def test_protected_route_requires_token(self):
        response = client.get("/watchlist")
        self.assertEqual(response.status_code, 401)

    def test_login_watchlist_and_saved_analysis_flow(self):
        username, token = self._register_and_login()
        headers = {"Authorization": f"Bearer {token}"}

        me = client.get("/auth/me", headers=headers)
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["data"]["username"], username)

        added = client.post("/watchlist", json={"ticker": " aapl "}, headers=headers)
        self.assertEqual(added.status_code, 201)
        self.assertEqual(added.json()["data"]["ticker"], "AAPL")
        self.assertTrue(added.json()["data"]["created_at"].endswith("Z"))
        duplicate = client.post("/watchlist", json={"ticker": "AAPL"}, headers=headers)
        self.assertEqual(duplicate.status_code, 409)

        saved = client.post(
            "/saved-analyses",
            json={"ticker": "aapl", "result": {"action": "HOLD"}},
            headers=headers,
        )
        self.assertEqual(saved.status_code, 201)
        self.assertTrue(saved.json()["data"]["created_at"].endswith("Z"))
        item_id = saved.json()["data"]["id"]
        history = client.get("/saved-analyses", headers=headers)
        self.assertTrue(any(item["id"] == item_id for item in history.json()["data"]))

        self.assertEqual(client.delete(f"/saved-analyses/{item_id}", headers=headers).status_code, 204)
        self.assertEqual(client.delete("/watchlist/AAPL", headers=headers).status_code, 204)

    def test_wrong_password_is_rejected(self):
        username = f"test_{uuid.uuid4().hex[:12]}"
        client.post("/auth/register", json={"username": username, "password": "correct-horse-42"})
        response = client.post("/auth/token", data={"username": username, "password": "wrong-password"})
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
