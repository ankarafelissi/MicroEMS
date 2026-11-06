"""Check forecast provenance and hand-calculated streaming error metrics."""

import copy
import unittest
from math import sqrt

from backend.evaluation import forecast_accuracy


class ForecastEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "forecast": [
                {"slot": 0, "time": "00:00", "pv_kw": 2, "load_kw": 10},
                {"slot": 1, "time": "00:15", "pv_kw": 4, "load_kw": 20},
            ],
            "executions": [],
            "revisions": [],
        }

    def test_no_actual_measurements_means_no_score(self):
        records, metrics = forecast_accuracy(self.state)
        self.assertEqual(records, [])
        for score in metrics.values():
            self.assertEqual(score["sample_count"], 0)
            self.assertIsNone(score["mae_kw"])
            self.assertIsNone(score["rmse_kw"])

    def test_scores_use_forecasts_issued_before_observation(self):
        self.state["executions"] = [
            {"slot": 0, "time": "00:00", "pv_kw": 3, "load_kw": 12},
            {"slot": 1, "time": "00:15", "pv_kw": 8, "load_kw": 16},
        ]
        self.state["revisions"] = [
            {
                "step": 0,
                "plan": {
                    "points": [
                        self.state["executions"][0],
                        {"slot": 1, "time": "00:15", "pv_kw": 6, "load_kw": 15},
                    ]
                },
            },
            {"step": 1, "plan": {"points": [self.state["executions"][1]]}},
        ]
        original = copy.deepcopy(self.state)
        records, metrics = forecast_accuracy(self.state)
        self.assertEqual(self.state, original)
        self.assertEqual(metrics["day_ahead_pv"]["mae_kw"], 2.5)
        self.assertAlmostEqual(metrics["day_ahead_pv"]["rmse_kw"], sqrt(8.5))
        self.assertEqual(metrics["day_ahead_load"]["mae_kw"], 3)
        self.assertAlmostEqual(metrics["day_ahead_load"]["rmse_kw"], sqrt(10))
        for source in ("pv", "load"):
            score = metrics[f"rolling_{source}"]
            self.assertEqual(score["sample_count"], 2)
            self.assertEqual(score["mae_kw"], 1.5)
            self.assertAlmostEqual(score["rmse_kw"], sqrt(2.5))
        self.assertEqual(records[1]["rolling_pv_kw"], 6)
        self.assertEqual(records[1]["rolling_issued_after_slot"], 0)

    def test_missing_old_revision_does_not_use_current_actual_as_prediction(self):
        self.state["executions"] = [
            {"slot": 1, "time": "00:15", "pv_kw": 8, "load_kw": 16},
        ]
        self.state["revisions"] = [
            {"step": 1, "plan": {"points": self.state["executions"]}},
        ]
        records, metrics = forecast_accuracy(self.state)
        self.assertEqual(metrics["day_ahead_pv"]["sample_count"], 1)
        self.assertEqual(metrics["rolling_pv"]["sample_count"], 0)
        self.assertIsNone(records[0]["rolling_pv_kw"])


if __name__ == "__main__":
    unittest.main()
