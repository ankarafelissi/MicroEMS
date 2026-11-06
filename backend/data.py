"""Build reproducible scenarios without exposing future observations to dispatch."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.predict import BASE_PATH, load_model, predict_load, predict_pv

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SLOTS = 96
DT = 0.25
DEMO_PV_CAPACITY_KW = 3.0
DEMO_LOAD_REFERENCE_KW = 3.0


@lru_cache(maxsize=1)
def load_source_data():
    pv = pd.read_csv(DATA_DIR / "gansu_pv.csv")
    load = pd.read_csv(DATA_DIR / "quanzhou_load.csv")
    load["times"] = pd.to_datetime(load["times"], errors="raise")
    load = load.sort_values("times").reset_index(drop=True)
    prices = pd.read_csv(DATA_DIR / "electricity_prices.csv")
    return pv, load, prices


def scenario_catalog():
    pv, load, _ = load_source_data()
    try:
        _, pv_checkpoint = load_model(
            "LSTMTransformerModel", str(BASE_PATH / "outputs/pv/best_model.pt")
        )
        _, load_checkpoint = load_model(
            "LSTMModel", str(BASE_PATH / "outputs/load/best_model.pt")
        )
    except (OSError, KeyError, RuntimeError) as error:
        raise ValueError(
            "Could not load the forecast models. Check both best_model.pt files in outputs."
        ) from error

    # PV training uses contiguous daily pairs: N samples cover N + 1 days.
    # Start after that whole dataset, including validation and test days.
    pv_day = sum(pv_checkpoint["split_samples"]) + 1
    load_day = len(load) // SLOTS - 1
    load_horizon = load_checkpoint["model_config"]["horizon"]
    # Missing test windows can only move this conservative boundary later.
    load_test_start = len(load) - load_checkpoint["split_samples"][2] - load_horizon + 1
    if (pv_day + 1) * SLOTS > len(pv):
        raise ValueError(
            "The PV dataset needs a complete evaluation day after the training range."
        )
    if load_day * SLOTS < max(load_test_start, 7 * SLOTS):
        raise ValueError(
            "The demand dataset has no complete evaluation day in the test range."
        )
    return {
        "pv_day_index": pv_day,
        "load_day_index": load_day,
        "forecast_method": "PV LSTM-Transformer + demand LSTM (trained weights)",
        "note": "PV and demand come from different regions and years, aligned into 96 intervals and scaled using historical reference levels.",
    }


def prepare_scenario():
    pv, load, prices = load_source_data()
    catalog = scenario_catalog()
    pv_start = catalog["pv_day_index"] * SLOTS
    load_start = catalog["load_day_index"] * SLOTS
    pv_history = pv.iloc[pv_start - 7 * SLOTS : pv_start]
    load_history = load.iloc[load_start - 7 * SLOTS : load_start]
    p = (
        pv["actual_power"]
        .iloc[pv_start - 7 * SLOTS : pv_start + SLOTS]
        .to_numpy(dtype=float)
    )
    l = (
        load["load"]
        .iloc[load_start - 7 * SLOTS : load_start + SLOTS]
        .to_numpy(dtype=float)
    )
    if (
        not np.isfinite(p).all()
        or not np.isfinite(l).all()
        or min(p.min(), l.min()) < 0
    ):
        raise ValueError(
            "The selected history or evaluation day contains missing, negative or infinite values."
        )
    # Inference and scaling see only history; target-day observations stay private.
    forecast_pv = predict_pv(pv_history)
    forecast_load = predict_load(load_history)
    pv_scale = DEMO_PV_CAPACITY_KW / max(float(p[:-SLOTS].max()), 1e-9)
    load_scale = DEMO_LOAD_REFERENCE_KW / max(float(l[:-SLOTS].max()), 1e-9)
    forecast_pv = np.minimum(forecast_pv * pv_scale, DEMO_PV_CAPACITY_KW)
    forecast_load = forecast_load * load_scale
    p = np.minimum(p * pv_scale, DEMO_PV_CAPACITY_KW)
    l = l * load_scale
    if len(prices) != 24:
        raise ValueError(
            "The tariff file must contain purchase and sale prices for all 24 hours."
        )
    buy = np.repeat(prices["buy_price"].to_numpy(dtype=float), 4)
    sell = np.repeat(prices["sell_price"].to_numpy(dtype=float), 4)
    if (
        not np.isfinite(buy).all()
        or not np.isfinite(sell).all()
        or min(buy.min(), sell.min()) < 0
    ):
        raise ValueError("Tariffs must be valid nonnegative numbers.")
    forecast = [
        {
            "slot": i,
            "time": f"{i // 4:02d}:{i % 4 * 15:02d}",
            "pv_kw": float(forecast_pv[i]),
            "load_kw": float(forecast_load[i]),
            "buy_price": float(buy[i]),
            "sell_price": float(sell[i]),
        }
        for i in range(SLOTS)
    ]
    return {
        "forecast": forecast,
        "pv_capacity_kw": DEMO_PV_CAPACITY_KW,
        "actual_pv_kw": p[-SLOTS:].tolist(),
        "actual_load_kw": l[-SLOTS:].tolist(),
        "source_info": {
            "forecast_method": "trained_lstm_transformer_and_lstm",
            "pv_model": "outputs/pv/best_model.pt (LSTM-Transformer)",
            "load_model": "outputs/load/best_model.pt (LSTM)",
            "pv_source": f"Gansu CSV, 96-point block {catalog['pv_day_index'] + 1}, first timestamp: {pv.iloc[pv_start, 0]}",
            "load_source": f"Quanzhou CSV, day {catalog['load_day_index'] + 1}, first timestamp: {load.iloc[load_start, 0]}",
            "alignment": "Offline data from different regions and years, aligned by interval; not a single site or calendar day.",
            "scaling": "Fixed dataset: seven-day historical PV and demand peaks are each scaled to 3 kW; PV is capped at 3 kW. This is scenario scaling, not a conversion of source units.",
            "forecast_note": "PV uses 17 features from the previous day; demand uses four features from the previous seven days. Neither reads target-day inputs. Intraday updates correct observed residuals without retraining.",
            "evaluation_split": "PV uses a day after the full training-script dataset; demand uses the last complete test day. Evaluation uses scaled scenario kW, not the units of the original model report.",
            "pv_scale": str(pv_scale),
            "load_scale": str(load_scale),
        },
    }


def rolling_forecast(original, step, actual_pv, actual_load, pv_capacity_kw=None):
    """Correct future predictions with the observed residual; current slot is exact."""
    remaining = []
    pv_error = actual_pv - original[step]["pv_kw"]
    load_error = actual_load - original[step]["load_kw"]
    for offset, point in enumerate(original[step:]):
        corrected = dict(point)
        weight = 1.0 if offset == 0 else 0.65 * np.exp(-offset / 8)
        corrected["pv_kw"] = max(0.0, float(point["pv_kw"] + weight * pv_error))
        if pv_capacity_kw is not None:
            corrected["pv_kw"] = min(corrected["pv_kw"], pv_capacity_kw)
        corrected["load_kw"] = max(0.0, float(point["load_kw"] + weight * load_error))
        remaining.append(corrected)
    return remaining
