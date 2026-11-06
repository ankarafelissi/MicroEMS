"""Forecast the next 24 hours from history using saved models, without training or database access."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

BASE_PATH = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_FEATURE_COLUMNS = [
    "mean_temperature_c",
    "mean_relative_humidity",
    "precipitation_mm",
    "load",
]
PV_FEATURE_COLUMNS = [
    "air_temperature",
    "azimuth",
    "cloud_opacity",
    "dew_point_temperature",
    "dhi",
    "dni",
    "ghi",
    "gti_fixed_tilt",
    "gti_tracking_tilt",
    "actual_power",
    "historical_power",
    "precipitable_water",
    "relative_humidity",
    "snow_depth",
    "surface_pressure",
    "wind_direction_10m",
    "wind_speed_10m",
]


class LSTMModel(nn.Module):
    """Match the load training architecture using configuration from the saved model."""

    def __init__(
        self, window_size, horizon, num_features, lstm_hid, lstm_layers, dropout
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            num_features,
            lstm_hid,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hid, horizon)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(self.dropout(output[:, -1, :]))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class LSTMTransformerModel(nn.Module):
    """Match the PV training architecture, including weight names and positional encoding layout."""

    def __init__(
        self,
        seq_len,
        num_features,
        pred_len,
        d_model,
        lstm_hid,
        lstm_layers,
        nhead,
        transformer_layers,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        self.lstm = nn.LSTM(
            d_model,
            lstm_hid,
            lstm_layers,
            batch_first=False,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.lstm_to_transformer = (
            nn.Linear(lstm_hid, d_model) if lstm_hid != d_model else nn.Identity()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, pred_len),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_projection(x).transpose(0, 1)
        x = self.dropout(self.pos_encoder(x))
        x, _ = self.lstm(x)
        x = self.transformer_encoder(self.lstm_to_transformer(x))
        return self.output_layer(x[-1, :, :])


@lru_cache(maxsize=2)
def load_model(model_type, model_path):
    """Load each model once; restart the process after replacing model files."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint["model_type"] != model_type:
        raise ValueError(f"Model type mismatch: {model_path}")
    if any(not column.isascii() for column in checkpoint["feature_columns"]):
        # Legacy checkpoints use the original fixed feature order with localized names.
        columns = (
            LOAD_FEATURE_COLUMNS if model_type == "LSTMModel" else PV_FEATURE_COLUMNS
        )
        if len(checkpoint["feature_columns"]) != len(columns):
            raise ValueError(f"Legacy model feature count mismatch: {model_path}")
        checkpoint["feature_columns"] = columns.copy()
    model_class = LSTMModel if model_type == "LSTMModel" else LSTMTransformerModel
    model = model_class(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, checkpoint


def _predict(history, model, checkpoint, window_size):
    """Normalize in training column order and restore predictions to original power units."""
    if len(history) < window_size:
        raise ValueError(
            f"At least {window_size} history rows are required; only {len(history)} are available."
        )
    missing = set(checkpoint["feature_columns"]) - set(history.columns)
    if missing:
        raise ValueError(
            "Required model input columns are missing. Check the input dataset."
        )
    values = history[checkpoint["feature_columns"]].iloc[-window_size:]
    values = values.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "The input window contains missing or infinite values. Provide continuous valid history."
        )

    normalization = checkpoint["normalization"]
    normalized = (values - np.asarray(normalization["feature_min"])) / np.asarray(
        normalization["feature_range"]
    )
    inputs = torch.as_tensor(normalized, dtype=torch.float32, device=DEVICE)
    with torch.inference_mode():
        prediction = model(inputs.unsqueeze(0)).cpu().numpy()[0]
    prediction = (
        prediction * normalization["target_range"] + normalization["target_min"]
    )
    if prediction.shape != (96,) or not np.isfinite(prediction).all():
        raise ValueError("The model must produce 96 valid power forecasts.")
    return np.maximum(prediction, 0)


def predict_load(history, model_path=BASE_PATH / "outputs/load/best_model.pt"):
    """Forecast load from the last seven days; times must be continuous at 15-minute intervals."""
    model, checkpoint = load_model("LSTMModel", str(model_path))
    missing = set(checkpoint["feature_columns"] + ["times"]) - set(history.columns)
    if missing:
        raise ValueError(
            "Required demand input columns are missing. Check the input dataset."
        )
    history = history.copy()
    history["times"] = pd.to_datetime(history["times"], errors="raise")
    history = history.sort_values("times")
    window_size = checkpoint["model_config"]["window_size"]
    times = history["times"].iloc[-window_size:]
    if (
        times.isna().any()
        or not times.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
    ):
        raise ValueError(
            "Demand timestamps must be continuous, unique and 15 minutes apart."
        )
    columns = checkpoint["feature_columns"]
    # Match training: forward-fill at most two consecutive history points before selecting the input window.
    history[columns] = (
        history[columns].astype(float).replace([np.inf, -np.inf], np.nan).ffill(limit=2)
    )
    return _predict(history, model, checkpoint, window_size)


def predict_pv(history, model_path=BASE_PATH / "outputs/pv/best_model.pt"):
    """Forecast PV from the last day of 17 features; preserve the training CSV row order."""
    model, checkpoint = load_model("LSTMTransformerModel", str(model_path))
    return _predict(history, model, checkpoint, checkpoint["model_config"]["seq_len"])


if __name__ == "__main__":
    pv_history = pd.read_csv(BASE_PATH / "data/gansu_pv.csv", nrows=500 * 96)
    load_history = pd.read_csv(BASE_PATH / "data/quanzhou_load.csv")
    for name, prediction in (
        ("PV", predict_pv(pv_history)),
        ("Load", predict_load(load_history)),
    ):
        print(
            f"{name}: {len(prediction)} points, "
            f"range={prediction.min():.2f}..{prediction.max():.2f} "
            "(source data unit)"
        )
