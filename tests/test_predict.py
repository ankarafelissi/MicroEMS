"""Check training and inference parity with saved weights, without training or database access."""

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecasting import predict


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pv = pd.read_csv(predict.BASE_PATH / "data/gansu_pv.csv", nrows=48000)
        cls.load = pd.read_csv(predict.BASE_PATH / "data/quanzhou_load.csv")

    def training_prediction(self, kind, history):
        """Extract only training class definitions to independently reproduce the original forward pass."""
        checkpoint = torch.load(
            predict.BASE_PATH / f"outputs/{kind}/best_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        source = Path(predict.BASE_PATH / f"forecasting/train_{kind}.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        classes = ast.Module(
            body=[node for node in tree.body if isinstance(node, ast.ClassDef)],
            type_ignores=[],
        )
        namespace = {
            "torch": torch,
            "nn": nn,
            "np": np,
            "Dataset": torch.utils.data.Dataset,
        }
        config = checkpoint["model_config"]
        aliases = {"transformer_layers": "NUM_TRANSFORMER_LAYERS"}
        namespace.update({aliases.get(k, k.upper()): v for k, v in config.items()})
        exec(compile(classes, str(source), "exec"), namespace)  # noqa: S102 -- Local class definitions only.
        model = namespace[checkpoint["model_type"]]().eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        window_size = config["window_size"] if kind == "load" else config["seq_len"]
        columns = (
            predict.LOAD_FEATURE_COLUMNS
            if kind == "load"
            else predict.PV_FEATURE_COLUMNS
        )
        values = history[columns].astype(float)
        if kind == "load":
            values = values.replace([np.inf, -np.inf], np.nan).ffill(limit=2)
        values = values.iloc[-window_size:].to_numpy()
        norm = checkpoint["normalization"]
        values = (values - np.array(norm["feature_min"])) / np.array(
            norm["feature_range"]
        )
        with torch.inference_mode():
            result = model(torch.tensor(values, dtype=torch.float32).unsqueeze(0))
        return np.maximum(
            result.numpy()[0] * norm["target_range"] + norm["target_min"], 0
        )

    def test_matches_training_networks(self):
        for kind, history, function in (
            ("pv", self.pv, predict.predict_pv),
            ("load", self.load, predict.predict_load),
        ):
            with self.subTest(model=kind):
                actual = function(history)
                expected = self.training_prediction(kind, history)
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=0.01)
                self.assertEqual(actual.shape, (96,))
                self.assertTrue(np.isfinite(actual).all())
                self.assertTrue((actual >= 0).all())
                np.testing.assert_array_equal(actual, function(history))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            predict.predict_pv(self.pv.tail(95))
        with self.assertRaises(ValueError):
            predict.predict_load(self.load.tail(671))
        with self.assertRaises(ValueError):
            predict.predict_load(self.load.drop(columns=["load"]))
        history = self.load.tail(672).copy()
        history.iloc[-1, history.columns.get_loc("times")] = history.iloc[-2]["times"]
        with self.assertRaises(ValueError):
            predict.predict_load(history)
        history = self.load.copy()
        history.loc[history.index[-3:], "load"] = np.nan
        with self.assertRaises(ValueError):
            predict.predict_load(history)

    def test_legacy_and_english_checkpoint_columns(self):
        for kind, model_type, columns in (
            ("load", "LSTMModel", predict.LOAD_FEATURE_COLUMNS),
            ("pv", "LSTMTransformerModel", predict.PV_FEATURE_COLUMNS),
        ):
            path = predict.BASE_PATH / f"outputs/{kind}/best_model.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            with self.subTest(model=kind, schema="legacy"):
                _, loaded = predict.load_model.__wrapped__(model_type, str(path))
                self.assertEqual(loaded["feature_columns"], columns)
                self.assertEqual(loaded["normalization"], checkpoint["normalization"])

            checkpoint["feature_columns"] = columns.copy()
            with (
                self.subTest(model=kind, schema="english"),
                patch("forecasting.predict.torch.load", return_value=checkpoint),
            ):
                _, loaded = predict.load_model.__wrapped__(model_type, str(path))
                self.assertEqual(loaded["feature_columns"], columns)

    def test_short_missing_load_gap(self):
        history = self.load.copy()
        history.loc[history.index[-2:], "load"] = np.nan
        np.testing.assert_allclose(
            predict.predict_load(history),
            self.training_prediction("load", history),
            rtol=1e-4,
            atol=0.01,
        )


if __name__ == "__main__":
    unittest.main()
