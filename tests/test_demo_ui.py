"""Exercise Streamlit forms and controls against the real API application."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from influx_support import repository_fixture
from streamlit.testing.v1 import AppTest

from backend.main import create_app

ROOT = Path(__file__).resolve().parents[1]


class DemoUiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(repository_fixture(self)))
        self.client.__enter__()

        def transport(method, url, json=None, timeout=None):
            path = "/" + url.split("://", 1)[1].split("/", 1)[1]
            response = self.client.request(method, path, json=json)
            return SimpleNamespace(
                ok=response.is_success,
                status_code=response.status_code,
                json=response.json,
            )

        self.patcher = patch("requests.request", side_effect=transport)
        self.patcher.start()
        self.app = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=20)

    def tearDown(self):
        self.patcher.stop()
        self.client.__exit__(None, None, None)

    def button(self, label):
        return next(item for item in self.app.button if item.label == label)

    def test_create_step_and_export_widgets(self):
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.info)
        self.assertTrue(
            any("MicroEMS-Guide/" in item.value for item in self.app.markdown)
        )
        self.assertEqual(
            {item.label for item in self.app.number_input},
            {
                "Grid import soft target (kW)",
                "Grid import hard limit (kW)",
                "Battery capacity (kWh)",
                "Charge / discharge limit (kW)",
                "Grid import capacity (kW)",
                "Grid export capacity (kW)",
            },
        )
        self.button("Generate day-ahead plan").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.metric), 7)
        visible_text = "\n".join(
            [item.value for item in self.app.markdown]
            + [item.value for item in self.app.caption]
            + [item.label for item in self.app.button]
            + [str(item.value) for item in self.app.table]
        )
        self.assertNotRegex(visible_text, r"[\u4e00-\u9fff]")
        scores = next(
            table.value for table in self.app.table if "MAE (kW)" in table.value.columns
        )
        self.assertTrue(scores["Samples"].eq(0).all())
        self.assertTrue(scores["MAE (kW)"].eq("—").all())
        simulation_id = self.app.session_state["simulation_id"]
        self.assertEqual(
            self.client.get(f"/api/simulations/{simulation_id}").json()["step"], 0
        )
        self.button("Execute next 15 minutes").click().run()
        self.assertFalse(self.app.exception)
        state = self.client.get(f"/api/simulations/{simulation_id}").json()
        self.assertEqual(state["step"], 1)
        self.assertEqual(len(state["executions"]), 1)
        scores = next(
            table.value for table in self.app.table if "MAE (kW)" in table.value.columns
        )
        self.assertTrue(scores["Samples"].eq(1).all())
        self.assertFalse(scores["MAE (kW)"].eq("—").any())
        self.assertNotIn("data", state["request"])
        self.app.run()
        self.assertEqual(
            self.client.get(f"/api/simulations/{simulation_id}").json()["step"], 1
        )

    def test_form_shows_constraint_conflict_without_crashing(self):
        self.app.run()
        next(
            item
            for item in self.app.checkbox
            if item.label == "Enable import hard limit"
        ).check()
        next(
            item
            for item in self.app.number_input
            if item.label == "Grid import hard limit (kW)"
        ).set_value(0.0)
        self.button("Generate day-ahead plan").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(
            any(
                "Operating requirements conflict" in item.value
                for item in self.app.error
            )
        )
        for error in self.app.error:
            self.assertNotRegex(error.value, r"[\u4e00-\u9fff]")
        self.assertEqual(self.client.get("/api/simulations").json(), [])

    def test_form_validates_initial_reserve_soc(self):
        self.app.run()
        next(
            item for item in self.app.slider if item.label == "Reserve SOC (%)"
        ).set_value(80)
        self.button("Generate day-ahead plan").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Initial SOC" in item.value for item in self.app.error))
        for error in self.app.error:
            self.assertNotRegex(error.value, r"[\u4e00-\u9fff]")


if __name__ == "__main__":
    unittest.main()
