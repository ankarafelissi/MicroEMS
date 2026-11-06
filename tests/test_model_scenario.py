"""Verify trained model integration and the evaluation-day data boundary."""

import unittest
from unittest.mock import patch

import numpy as np

from backend.data import (
    DEMO_PV_CAPACITY_KW,
    SLOTS,
    load_source_data,
    prepare_scenario,
    scenario_catalog,
)
from forecasting.predict import BASE_PATH, load_model, predict_load, predict_pv


class ModelScenarioTests(unittest.TestCase):
    def test_forecast_matches_models_with_identical_scaling(self):
        pv, load, _ = load_source_data()
        catalog = scenario_catalog()
        scenario = prepare_scenario()
        pv_start = catalog["pv_day_index"] * SLOTS
        load_start = catalog["load_day_index"] * SLOTS
        expected_pv = np.minimum(
            predict_pv(pv.iloc[pv_start - SLOTS : pv_start])
            * float(scenario["source_info"]["pv_scale"]),
            DEMO_PV_CAPACITY_KW,
        )
        expected_load = predict_load(load.iloc[load_start - 7 * SLOTS : load_start])
        expected_load *= float(scenario["source_info"]["load_scale"])
        np.testing.assert_allclose(
            [p["pv_kw"] for p in scenario["forecast"]], expected_pv
        )
        np.testing.assert_allclose(
            [p["load_kw"] for p in scenario["forecast"]], expected_load
        )
        self.assertEqual(
            scenario["source_info"]["forecast_method"],
            "trained_lstm_transformer_and_lstm",
        )
        # The selected PV day follows every day used by its training script.
        _, checkpoint = load_model(
            "LSTMTransformerModel", str(BASE_PATH / "outputs/pv/best_model.pt")
        )
        self.assertEqual(catalog["pv_day_index"], sum(checkpoint["split_samples"]) + 1)

    def test_future_weather_and_measurements_cannot_change_forecast(self):
        pv, load, prices = load_source_data()
        catalog = scenario_catalog()
        scenario = prepare_scenario()
        pv_start = catalog["pv_day_index"] * SLOTS
        load_start = catalog["load_day_index"] * SLOTS
        changed_pv, changed_load = pv.copy(), load.copy()
        for frame, start in ((changed_pv, pv_start), (changed_load, load_start)):
            numeric = frame.select_dtypes(include="number").columns
            frame.loc[start : start + SLOTS - 1, numeric] = 10000
        with patch(
            "backend.data.load_source_data",
            return_value=(changed_pv, changed_load, prices),
        ):
            changed = prepare_scenario()
        self.assertEqual(scenario["forecast"], changed["forecast"])
        self.assertNotEqual(scenario["actual_load_kw"], changed["actual_load_kw"])

    def test_missing_unseen_pv_day_is_reported(self):
        pv, load, prices = load_source_data()
        start = scenario_catalog()["pv_day_index"] * SLOTS
        with (
            patch(
                "backend.data.load_source_data",
                return_value=(pv.iloc[:start], load, prices),
            ),
            self.assertRaisesRegex(ValueError, "after the training range"),
        ):
            prepare_scenario()


if __name__ == "__main__":
    unittest.main()
