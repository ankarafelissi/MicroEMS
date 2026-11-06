"""Failure injection for atomic publication and native time-series values."""

import copy
import json
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from influx_support import FakeInfluxClient
from requests.exceptions import ConnectionError

from backend.repository import MEASUREMENT, SimulationRepository
from backend.schemas import Conflict
from backend.telemetry import DISPATCH_MEASUREMENT, METRICS_MEASUREMENT


def state_fixture():
    records = [
        {
            "slot": slot,
            "time": f"00:{slot * 15:02d}",
            "pv_kw": 1.0,
            "load_kw": 2.0,
            "charge_kw": 0.0,
            "discharge_kw": 0.2,
            "import_kw": 0.8,
            "export_kw": 0.0,
            "curtail_kw": 0.0,
            "soc": 0.49 - slot * 0.01,
            "buy_price": 0.5,
            "sell_price": 0.3,
            "cost_yuan": 0.1,
            "target_excess_kw": 0.0,
        }
        for slot in range(2)
    ]
    plan = {"from_step": 0, "points": records}
    return {
        "id": "native-series",
        "created_at": "2026-10-20T15:41:02+01:00",
        "step": 0,
        "status": "ready",
        "soc": 0.5,
        "forecast": records,
        "day_ahead": plan,
        "latest_plan": plan,
        "executions": [],
        "revisions": [],
        "actual_pv_kw": [99.0, 98.0],
        "actual_load_kw": [97.0, 96.0],
    }


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeInfluxClient()
        self.repository = SimulationRepository(self.client)
        self.state = state_fixture()
        self.repository.create(self.state)

    def published_points(self, measurement):
        generation = self.repository._find(self.state["id"])["generation"]
        return [
            {"time": timestamp, **dict(tags), **fields}
            for (name, tags, timestamp), fields in self.client.points.items()
            if name == measurement and dict(tags)["generation"] == generation
        ]

    def advance(self):
        self.state["executions"] = copy.deepcopy(self.state["day_ahead"]["points"])
        self.state.update(step=2, status="completed", soc=0.48)

    def test_simulation_clock_and_native_fields_match_snapshot(self):
        self.assertFalse(
            any(
                point["phase"] == "executed"
                for point in self.published_points(DISPATCH_MEASUREMENT)
            )
        )
        self.advance()
        self.repository.save(self.state)
        points = [
            p
            for p in self.published_points(DISPATCH_MEASUREMENT)
            if p["phase"] == "executed"
        ]
        self.assertEqual(len(points), 2)
        timestamps = [datetime.fromisoformat(p["time"]) for p in points]
        self.assertEqual(timestamps[0].isoformat(), "2026-10-20T00:00:00+00:00")
        self.assertEqual((timestamps[1] - timestamps[0]).total_seconds(), 900)
        for point, actual in zip(points, self.state["executions"]):
            for field in ("pv_kw", "load_kw", "soc", "buy_price", "cost_yuan"):
                self.assertEqual(point[field], actual[field])
            self.assertAlmostEqual(point["battery_kw"], 0.2)
            self.assertEqual(point["battery_mode"], 2)
            self.assertEqual(point["grid_mode"], 1)
        metrics = self.published_points(METRICS_MEASUREMENT)[0]
        self.assertEqual(metrics["cost_yuan"], 0.2)
        self.assertEqual(metrics["imported_kwh"], 0.4)
        self.assertEqual(metrics["step"], 2)

    def test_partial_projection_is_not_published_and_retry_is_idempotent(self):
        original = self.repository.get(self.state["id"])
        generation = self.repository._find(self.state["id"])["generation"]
        write = self.client.write_points
        self.advance()

        def partial_write(points):
            write(points[:1])
            raise ConnectionError("response lost after a partial batch")

        with (
            patch.object(self.client, "write_points", side_effect=partial_write),
            self.assertRaises(ConnectionError),
        ):
            self.repository.save(self.state)
        self.assertEqual(self.repository.get(self.state["id"]), original)
        self.assertEqual(
            self.repository._find(self.state["id"])["generation"], generation
        )
        self.repository.save(self.state)
        points = self.published_points(DISPATCH_MEASUREMENT)
        self.assertEqual(len(points), 6)
        self.assertEqual(self.repository.get(self.state["id"]), self.state)

    def test_lost_commit_response_is_confirmed_without_duplicate_execution(self):
        write = self.client.write_points
        stale = copy.deepcopy(self.state)
        self.advance()

        def lost_response(points):
            result = write(points)
            if points[0]["measurement"] == MEASUREMENT:
                raise ConnectionError("commit succeeded, response lost")
            return result

        with patch.object(self.client, "write_points", side_effect=lost_response):
            self.repository.save(self.state)
        self.assertEqual(self.state["version"], 1)
        self.assertEqual(self.repository.get(self.state["id"])["step"], 2)
        with self.assertRaises(Conflict):
            self.repository.save(stale)

    def test_backfill_keeps_existing_state_and_version(self):
        original = self.repository.get(self.state["id"])
        for (name, _, _), fields in self.client.points.items():
            if name == MEASUREMENT:
                del fields["generation"]
        self.repository.repair_time_series()
        self.assertEqual(self.repository.get(self.state["id"]), original)
        self.assertEqual(len(self.published_points(DISPATCH_MEASUREMENT)), 4)

    def test_dashboard_filters_every_panel_by_plan_and_published_generation(self):
        root = Path(__file__).resolve().parents[1]
        dashboard = json.loads(
            (root / "deploy/grafana/dashboards/ems.json").read_text()
        )
        variables = {v["name"]: v for v in dashboard["templating"]["list"]}
        self.assertEqual(variables["generation"]["refresh"], 2)
        self.assertIn('FROM "ems_simulations"', variables["generation"]["query"])
        for panel in dashboard["panels"]:
            for target in panel.get("targets", []):
                self.assertIn("\"simulation_id\" = '${simulation_id}'", target["query"])
                self.assertIn("\"generation\" = '${generation}'", target["query"])


if __name__ == "__main__":
    unittest.main()
