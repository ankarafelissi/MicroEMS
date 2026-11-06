"""Physical, temporal and API invariants for the two-stage EMS demo."""

import copy
import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import gurobipy as gp
from fastapi.testclient import TestClient
from influx_support import repository_fixture
from requests.exceptions import ConnectionError

from backend.data import DT, prepare_scenario, rolling_forecast, scenario_catalog
from backend.main import create_app
from backend.optimizer import solve_dispatch
from backend.schemas import Conflict, CreateSimulation, OptimizationError
from backend.service import SimulationService


def tiny_forecast(load=8.0):
    return [
        {
            "slot": i,
            "time": f"00:{i * 15:02d}",
            "pv_kw": 0.0,
            "load_kw": load,
            "buy_price": 0.3,
            "sell_price": 0.5,
        }
        for i in range(2)
    ]


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.repository = repository_fixture(self)
        self.service = SimulationService(self.repository)

    def tearDown(self):
        self.service.shutdown()

    def test_storage_outage_stops_worker_and_recovery_pauses_without_advancing(self):
        state = self.service.create(CreateSimulation())
        saved = self.repository.get(state["id"])
        saved["status"] = "running"
        self.repository.save(saved)
        with (
            patch.object(
                self.repository.client, "query", side_effect=ConnectionError("offline")
            ),
            self.assertLogs("backend.service", level="ERROR"),
        ):
            self.service._run(state["id"], 0, Event())
        recovered = self.service.get(state["id"])
        self.assertEqual(recovered["status"], "paused")
        self.assertEqual(recovered["step"], 0)
        self.assertEqual(recovered["executions"], [])
        self.assertEqual(recovered["soc"], state["soc"])

    def test_soft_target_allows_excess_but_hard_limit_rejects(self):
        request = CreateSimulation.model_validate(
            {
                "enterprise": {"import_target_kw": 1},
                "equipment": {"grid_import_capacity_kw": 10},
            }
        )
        result = solve_dispatch(tiny_forecast(), request, 0.5)
        self.assertTrue(any(p["target_excess_kw"] > 0 for p in result["points"]))
        request.enterprise.import_hard_limit_kw = 5
        with self.assertRaises(OptimizationError) as caught:
            solve_dispatch(tiny_forecast(), request, 0.5)
        self.assertEqual(caught.exception.detail["code"], "infeasible")

    def test_last_slot_handles_float_roundoff_at_terminal_soc(self):
        forecast = tiny_forecast(load=2)[:1]
        forecast[0].update(slot=95, time="23:45")
        plan = solve_dispatch(forecast, CreateSimulation(), 0.49999999999999994)
        self.assertAlmostEqual(plan["points"][0]["soc"], 0.5, places=6)
        self.assertAlmostEqual(plan["points"][0]["import_kw"], 2, places=5)

    def test_solver_reuses_license_across_request_threads(self):
        solve_dispatch(tiny_forecast(load=2), CreateSimulation(), 0.5)
        with (
            patch(
                "backend.optimizer.gp.Env", side_effect=AssertionError("new session")
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            plans = list(
                executor.map(
                    lambda _: solve_dispatch(
                        tiny_forecast(load=2), CreateSimulation(), 0.5
                    ),
                    range(2),
                )
            )
        self.assertTrue(all(plan["solver_status"] == "optimal" for plan in plans))

    def test_license_failure_is_reported_without_exposing_credentials(self):
        with (
            patch("backend.optimizer._environment", None),
            patch(
                "backend.optimizer.gp.Env",
                side_effect=gp.GurobiError(10009, "private license details"),
            ),
            self.assertRaises(OptimizationError) as caught,
        ):
            solve_dispatch(tiny_forecast(), CreateSimulation(), 0.5)
        self.assertEqual(caught.exception.detail["code"], "solver_failed")
        self.assertIn("Gurobi", caught.exception.detail["message"])
        self.assertNotIn("private license details", str(caught.exception.detail))

    def test_solver_timeout_does_not_execute_a_step(self):
        state = self.service.create(CreateSimulation())
        with (
            patch("backend.optimizer.monotonic", side_effect=[0, 11]),
            self.assertRaises(OptimizationError) as caught,
        ):
            self.service.step(state["id"], 0)
        self.assertEqual(caught.exception.detail["code"], "solver_failed")
        saved = self.service.get(state["id"])
        self.assertEqual(saved["step"], 0)
        self.assertEqual(saved["soc"], state["soc"])
        self.assertEqual(saved["executions"], [])

    def test_peak_mode_minimizes_peak_before_cost_without_weights(self):
        request = CreateSimulation.model_validate(
            {
                "enterprise": {"import_target_kw": None, "reserve_soc": 0},
                "equipment": {
                    "efficiency": 1,
                    "charge_limit_kw": 3,
                    "discharge_limit_kw": 3,
                    "grid_export_capacity_kw": 0,
                },
            }
        )
        forecast = tiny_forecast()
        forecast[0].update(load_kw=1, buy_price=0.1)
        forecast[1].update(load_kw=4, buy_price=10)
        cost = solve_dispatch(forecast, request, 0.5)["points"]
        request.enterprise.objective = "peak"
        peak = solve_dispatch(forecast, request, 0.5)["points"]
        self.assertAlmostEqual(max(p["import_kw"] for p in cost), 4, places=5)
        self.assertAlmostEqual(max(p["import_kw"] for p in peak), 2.5, places=5)
        self.assertLess(
            sum(p["cost_yuan"] for p in cost),
            sum(p["cost_yuan"] for p in peak),
        )
        # Once the best peak is fixed, use cheap energy to minimize the bill.
        self.assertAlmostEqual(peak[0]["charge_kw"], 1.5, places=5)

        request.enterprise.objective = "cost"
        request.enterprise.import_target_kw = 2.5
        soft = solve_dispatch(forecast, request, 0.5)["points"]
        self.assertLessEqual(max(p["target_excess_kw"] for p in soft), 1e-5)

    def test_peak_mode_completes_full_day(self):
        state = self.service.create(
            CreateSimulation.model_validate({"enterprise": {"objective": "peak"}})
        )
        for step in range(96):
            with self.subTest(step=step):
                state = self.service.step(state["id"], step)
                self.assertTrue(state["metrics"]["reserve_satisfied"])
                self.assertTrue(state["metrics"]["hard_limit_satisfied"])
        self.assertEqual(state["status"], "completed")
        self.assertGreaterEqual(state["soc"], 0.5 - 1e-5)

    def test_power_soc_and_direction_constraints_with_export_arbitrage_prices(self):
        request = CreateSimulation()
        points = tiny_forecast(load=1.0)
        points[1]["pv_kw"] = 4
        result = solve_dispatch(points, request, 0.5)
        soc = 0.5
        b = request.equipment
        for p in result["points"]:
            self.assertAlmostEqual(
                p["pv_kw"] + p["import_kw"] + p["discharge_kw"],
                p["load_kw"] + p["export_kw"] + p["charge_kw"] + p["curtail_kw"],
                places=5,
            )
            self.assertLess(p["charge_kw"] * p["discharge_kw"], 1e-6)
            self.assertLess(p["import_kw"] * p["export_kw"], 1e-6)
            soc += (
                DT
                * (p["charge_kw"] * b.efficiency - p["discharge_kw"] / b.efficiency)
                / b.capacity_kwh
            )
            self.assertAlmostEqual(soc, p["soc"], places=6)
        self.assertGreaterEqual(soc, b.terminal_soc - 1e-6)

    def test_full_day_preserves_plan_and_only_executes_one_step(self):
        state = self.service.create(CreateSimulation())
        ahead = copy.deepcopy(state["day_ahead"])
        simulation_id = state["id"]
        for step in range(96):
            state = self.service.step(simulation_id, step)
            self.assertEqual(state["step"], step + 1)
            self.assertEqual(len(state["executions"]), step + 1)
            self.assertEqual(len(state["latest_plan"]["points"]), 96 - step)
            self.assertGreaterEqual(state["soc"], 0.3 - 1e-5)
            self.assertLessEqual(state["soc"], 0.9 + 1e-5)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["day_ahead"], ahead)
        self.assertEqual(len(state["revisions"]), 96)
        self.assertGreaterEqual(state["soc"], 0.5 - 1e-5)
        self.assertTrue(state["metrics"]["hard_limit_satisfied"])
        self.assertEqual(
            state["prediction_metrics"]["day_ahead_pv"]["sample_count"], 96
        )
        self.assertEqual(
            state["prediction_metrics"]["rolling_load"]["sample_count"], 96
        )
        self.assertEqual(len(state["prediction_evaluations"]), 96)
        self.assertEqual(
            state["prediction_evaluations"][1]["rolling_issued_after_slot"], 0
        )
        private = self.repository.get(simulation_id)
        for i, p in enumerate(state["executions"]):
            self.assertAlmostEqual(p["pv_kw"], private["actual_pv_kw"][i])
            self.assertAlmostEqual(p["load_kw"], private["actual_load_kw"][i])
        self.assertTrue(
            any(
                abs(p["pv_kw"] - state["forecast"][i]["pv_kw"]) > 1e-6
                for i, p in enumerate(state["executions"])
            )
        )

    def test_unrevealed_future_does_not_influence_next_action(self):
        one = self.service.create(CreateSimulation())
        two = self.service.create(CreateSimulation())
        hidden = self.repository.get(two["id"])
        hidden["actual_load_kw"][1:] = [1000] * 95
        hidden["actual_pv_kw"][1:] = [1000] * 95
        self.repository.save(hidden)
        one = self.service.step(one["id"], 0)
        two = self.service.step(two["id"], 0)
        self.assertEqual(one["executions"], two["executions"])
        self.assertEqual(one["latest_plan"], two["latest_plan"])
        self.assertNotIn("actual_load_kw", two)
        self.assertNotIn("actual_pv_kw", two)
        self.assertEqual(one["prediction_metrics"], two["prediction_metrics"])
        self.assertEqual(one["prediction_evaluations"], two["prediction_evaluations"])

    def test_infeasible_step_does_not_advance_or_change_soc(self):
        state = self.service.create(CreateSimulation())
        hidden = self.repository.get(state["id"])
        hidden["actual_load_kw"][0] = 100
        self.repository.save(hidden)
        with self.assertRaises(OptimizationError):
            self.service.step(state["id"], 0)
        saved = self.service.get(state["id"])
        self.assertEqual(saved["status"], "blocked")
        self.assertEqual(saved["step"], 0)
        self.assertEqual(saved["soc"], state["soc"])
        self.assertEqual(saved["executions"], [])

    def test_duplicate_concurrent_steps_cannot_execute_twice(self):
        state = self.service.create(CreateSimulation())

        def step_once():
            try:
                self.service.step(state["id"], 0)
                return "ok"
            except Conflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: step_once(), range(2)))
        self.assertCountEqual(results, ["ok", "conflict"])
        self.assertEqual(self.service.get(state["id"])["step"], 1)

    def test_reload_and_restart_preserve_state(self):
        state = self.service.create(CreateSimulation())
        state = self.service.step(state["id"], 0)
        for _ in range(3):
            self.assertEqual(self.service.get(state["id"])["step"], 1)
        hidden = self.repository.get(state["id"])
        hidden["status"] = "running"
        self.repository.save(hidden)
        restored = SimulationService(self.repository)
        result = restored.get(state["id"])
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["executions"], state["executions"])
        self.assertEqual(result["prediction_metrics"], state["prediction_metrics"])
        self.assertEqual(
            result["prediction_evaluations"], state["prediction_evaluations"]
        )

    def test_automatic_execution_and_pause(self):
        state = self.service.create(CreateSimulation())
        self.service.start(state["id"], 0.2)
        deadline = time.monotonic() + 5
        while self.service.get(state["id"])["step"] < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        paused = self.service.pause(state["id"])
        self.assertGreater(paused["step"], 0)
        self.assertEqual(paused["status"], "paused")
        time.sleep(0.35)
        self.assertEqual(self.service.get(state["id"])["step"], paused["step"])

    def test_forecasts_and_scaling_do_not_look_at_target_day(self):
        from backend.data import load_source_data

        pv, load, prices = load_source_data()
        catalog = scenario_catalog()
        expected = prepare_scenario()
        changed_pv, changed_load = pv.copy(), load.copy()
        pv_start, load_start = (
            catalog["pv_day_index"] * 96,
            catalog["load_day_index"] * 96,
        )
        changed_pv.loc[pv_start : pv_start + 95, "actual_power"] = 10000
        changed_load.loc[load_start : load_start + 95, "load"] = 10000
        with patch(
            "backend.data.load_source_data",
            return_value=(changed_pv, changed_load, prices),
        ):
            changed = prepare_scenario()
        self.assertEqual(expected["forecast"], changed["forecast"])
        self.assertNotEqual(expected["actual_load_kw"], changed["actual_load_kw"])

    def test_rolling_forecast_starts_with_current_measurement(self):
        result = rolling_forecast(tiny_forecast(), 0, 0.2, 9)
        self.assertEqual(result[0]["pv_kw"], 0.2)
        self.assertEqual(result[0]["load_kw"], 9)
        self.assertGreater(result[1]["load_kw"], 8)
        clipped = rolling_forecast(tiny_forecast(), 0, 3, 9, pv_capacity_kw=2)
        self.assertLessEqual(max(p["pv_kw"] for p in clipped), 2)

    def test_repository_rejects_stale_snapshot(self):
        state = self.service.create(CreateSimulation())
        first = self.repository.get(state["id"])
        stale = self.repository.get(state["id"])
        self.repository.save(first)
        with self.assertRaises(Conflict):
            self.repository.save(stale)

    def test_api_contracts_and_validation(self):
        with TestClient(create_app(self.repository)) as client:
            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["storage"], "influxdb")
            self.assertEqual(health.json()["solver"], "gurobi")
            self.assertEqual(client.get("/api/simulations/missing").status_code, 404)
            removed_inputs = [
                {"data": {"day_index": 60}},
                {"data": {"pv_capacity_kw": 3}},
                {"data": {"load_reference_kw": 3}},
                {"enterprise": {"excess_penalty": 2}},
                {"enterprise": {"plan_deviation_penalty": 0.03}},
                {"enterprise": {"peak_penalty": 2}},
            ]
            for payload in removed_inputs:
                with self.subTest(removed_input=payload):
                    response = client.post("/api/simulations", json=payload)
                    self.assertEqual(response.status_code, 422)
            self.assertEqual(
                client.post(
                    "/api/simulations", json={"enterprise": {"reserve_soc": 0.8}}
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    "/api/simulations", json={"enterprise": {"import_target_kw": "NaN"}}
                ).status_code,
                422,
            )
            created = client.post("/api/simulations", json={})
            self.assertEqual(created.status_code, 201, created.text)
            state = created.json()
            route = f"/api/simulations/{state['id']}/step"
            self.assertNotIn("actual_load_kw", state)
            self.assertEqual(
                client.post(route, json={"expected_step": 0}).status_code, 200
            )
            self.assertEqual(
                client.post(route, json={"expected_step": 0}).status_code, 409
            )
            bad = client.post(
                "/api/simulations", json={"enterprise": {"import_hard_limit_kw": 0}}
            )
            self.assertEqual(bad.status_code, 409)
            self.assertEqual(bad.json()["detail"]["code"], "infeasible")

    def test_api_errors_and_metadata_are_english(self):
        with TestClient(create_app(self.repository)) as client:
            for path in ("/api/scenarios", "/api/simulations/missing", "/openapi.json"):
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertNotRegex(
                        json.dumps(response.json(), ensure_ascii=False),
                        r"[\u3400-\u9fff]",
                    )
            for payload in (
                {"enterprise": {"reserve_soc": 0.8}},
                {"enterprise": {"objective": "unsupported input"}},
                {"unknown_parameter": "unsupported input"},
                {"enterprise": {"import_hard_limit_kw": 0}},
            ):
                with self.subTest(payload=payload):
                    response = client.post("/api/simulations", json=payload)
                    self.assertIn(response.status_code, (409, 422))
                    self.assertNotRegex(
                        json.dumps(response.json(), ensure_ascii=False),
                        r"[\u3400-\u9fff]",
                    )
            created = client.post("/api/simulations", json={})
            self.assertEqual(created.status_code, 201)
            self.assertNotRegex(
                json.dumps(created.json(), ensure_ascii=False), r"[\u3400-\u9fff]"
            )


if __name__ == "__main__":
    unittest.main()
