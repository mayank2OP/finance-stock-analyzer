import time
import unittest

from job_service import create_analysis_job, get_analysis_job


class JobServiceTests(unittest.TestCase):
    def test_job_reaches_completed_state(self):
        job = create_analysis_job("AAPL", lambda ticker: {"ticker": ticker, "action": "HOLD"})
        completed = None
        for _ in range(50):
            completed = get_analysis_job(job["job_id"])
            if completed and completed["status"] == "completed":
                break
            time.sleep(0.01)

        self.assertIsNotNone(completed)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["progress_percent"], 100)
        self.assertEqual(completed["result"]["ticker"], "AAPL")

    def test_missing_job_returns_none(self):
        self.assertIsNone(get_analysis_job("not-a-real-job"))

    def test_job_is_visible_only_to_its_owner(self):
        job = create_analysis_job("MSFT", lambda ticker: {"ticker": ticker}, owner_id=101)
        self.assertIsNotNone(get_analysis_job(job["job_id"], owner_id=101))
        self.assertIsNone(get_analysis_job(job["job_id"], owner_id=202))
        self.assertNotIn("_owner_id", job)


if __name__ == "__main__":
    unittest.main()
