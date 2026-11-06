"""Train on CSV temperature, humidity, precipitation and load time series."""

# ===================== Imports =====================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# ===================== Hyperparameters =====================
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "quanzhou_load.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "load"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WINDOW_SIZE = 96 * 7  # Input: past seven days = 672 points
HORIZON = 96  # Output: next day = 96 points
NUM_FEATURES = 4  # Features: temperature, humidity, precipitation and load
LSTM_HID = 128
LSTM_LAYERS = 2
DROPOUT = 0.2
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 40
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Set random seeds =====================
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"[*] Using device: {DEVICE}")

# ===================== 1. Data loading and preprocessing =====================
print("\n[Step 1] Data loading and preprocessing")
print("-" * 50)

df = pd.read_csv(DATA_PATH)
df["times"] = pd.to_datetime(df["times"])
df = df.sort_values("times").reset_index(drop=True)
input_columns = [
    "mean_temperature_c",
    "mean_relative_humidity",
    "precipitation_mm",
    "load",
]
df[input_columns] = df[input_columns].astype(float)
print(f"  Raw data rows: {len(df)}")

# Extract four features
features = pd.DataFrame()
features["temp"] = df["mean_temperature_c"].values
features["humidity"] = df["mean_relative_humidity"].values
features["precipitation"] = df["precipitation_mm"].values
features["load"] = df["load"].values

# Forward-fill inputs from past values for at most two points (30 minutes); keep targets as observed.
features = features.replace([np.inf, -np.inf], np.nan)
observed_load = features["load"].to_numpy(copy=True)
features = features.ffill(limit=2)
raw_data = features.values

# Set the exclusive training target boundary; fit only on the training period.
target_points = len(raw_data) - WINDOW_SIZE
train_end = WINDOW_SIZE + int(target_points * TRAIN_RATIO)
val_end = train_end + int(target_points * VAL_RATIO)

# Min-max normalization: reuse training parameters for validation/test without refitting or clipping.
data_min = np.nanmin(raw_data[:train_end], axis=0)
data_max = np.nanmax(raw_data[:train_end], axis=0)
data_range = data_max - data_min
data_range[data_range < 1e-8] = 1.0
normalized = (raw_data - data_min) / data_range
normalized_load = (observed_load - data_min[-1]) / data_range[-1]
print(
    "  Normalization fitted on the training period only; metrics and load plots use original units"
)

# Build time-series samples
X, y, sample_starts = [], [], []
total_need = WINDOW_SIZE + HORIZON
for i in range(len(normalized) - total_need + 1):
    history = normalized[i : i + WINDOW_SIZE]
    future_load = normalized_load[i + WINDOW_SIZE : i + total_need]
    if not np.isfinite(history).all() or not np.isfinite(future_load).all():
        continue
    X.append(history)
    y.append(future_load)
    sample_starts.append(i)
X = np.array(X)
y = np.array(y)

n = len(X)
# Split by target period; exclude 96-point target windows that cross split boundaries.
# Keep original row indices after skipping missing windows to preserve split boundaries.
target_start = np.array(sample_starts, dtype=int) + WINDOW_SIZE
target_end = target_start + HORIZON  # Exclusive end

train_idx = np.flatnonzero(target_end <= train_end)
val_idx = np.flatnonzero((target_start >= train_end) & (target_end <= val_end))
test_idx = np.flatnonzero(target_start >= val_end)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(
    f"  Sequence samples: {n}, Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}"
)
print(
    f"  Excluded samples with missing values: {max(0, len(normalized) - total_need + 1) - n}"
)
print(
    f"  Excluded samples crossing split boundaries: {n - len(train_idx) - len(val_idx) - len(test_idx)}"
)


# Create datasets and data loaders
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(
    TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
)

# Data diagnostics
batch = next(iter(train_loader))
print(f"  X shape: {batch[0].shape}, y shape: {batch[1].shape}")

# ===================== 2. LSTM model definition =====================
print("\n[Step 2] Build the LSTM model")
print("-" * 50)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=NUM_FEATURES,
            hidden_size=LSTM_HID,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(LSTM_HID, HORIZON)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)
        return output


model = LSTMModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

n_params = sum(p.numel() for p in model.parameters())
print(f"  LSTM parameter count: {n_params:,}")
print(f"  hidden size: {LSTM_HID}, layers: {LSTM_LAYERS}, learning rate: {LR}")

# ===================== 3. Model training =====================
print("\n[Step 3] Model training")
print("-" * 50)

train_losses = []
val_losses = []
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # Training
    model.train()
    batch_losses = []
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    avg_train_loss = np.mean(batch_losses)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_batch_losses.append(loss.item())

    avg_val_loss = np.mean(val_batch_losses)
    val_losses.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = float(avg_val_loss)
        checkpoint = {
            "model_type": "LSTMModel",
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "model_config": {
                "window_size": WINDOW_SIZE,
                "horizon": HORIZON,
                "num_features": NUM_FEATURES,
                "lstm_hid": LSTM_HID,
                "lstm_layers": LSTM_LAYERS,
                "dropout": DROPOUT,
            },
            "feature_columns": input_columns,
            "normalization": {
                "feature_min": data_min.tolist(),
                "feature_range": data_range.tolist(),
                "target_min": float(data_min[-1]),
                "target_range": float(data_range[-1]),
            },
            "best_epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "sampling_minutes": 15,
            "power_unit": "source_data_unit",
            "data_file": str(DATA_PATH),
            "split_samples": [len(y_train), len(y_val), len(y_test)],
            "training_config": {
                "seed": SEED,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            },
        }
        torch.save(checkpoint, OUTPUT_DIR / "best_model.pt")

    print(
        f"  Epoch {epoch + 1:2d}/{EPOCHS} - training loss: {avg_train_loss:.6f}, validation loss: {avg_val_loss:.6f}"
    )

print("\n  Training complete!")
model.load_state_dict(checkpoint["model_state_dict"])
print(f"  Loaded best weights from epoch {checkpoint['best_epoch']}")

# ===================== 4. Model evaluation =====================
print("\n[Step 4] Model evaluation")
print("-" * 50)

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(DEVICE)
        outputs = model(batch_X)
        all_preds.append(outputs.cpu().numpy())
        all_true.append(batch_y.numpy())

preds = np.concatenate(all_preds, axis=0)
true = np.concatenate(all_true, axis=0)

# Restore original units for the load column only
load_min = data_min[-1]
load_range = data_range[-1]
preds_denorm = preds * load_range + load_min
true_denorm = true * load_range + load_min

# Compute metrics
y_true_flat = true_denorm.flatten()
y_pred_flat = preds_denorm.flatten()

rmse = np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2))

numerator = np.abs(y_pred_flat - y_true_flat)
denominator = np.abs(y_pred_flat) + np.abs(y_true_flat)
denominator[denominator == 0] = 1e-8
smape = 200 * np.mean(numerator / denominator)

print(f"  Test RMSE: {rmse:.2f} kW")
print(f"  Test SMAPE: {smape:.2f} p.u.")
checkpoint["test_metrics"] = {"rmse": float(rmse), "smape_percent": float(smape)}
torch.save(checkpoint, OUTPUT_DIR / "best_model.pt")

# ===================== 5. Save two key figures =====================
print("\n[Step 5] Generate plots")
print("-" * 50)

# Figure 1: training loss curves
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(
    range(1, EPOCHS + 1),
    train_losses,
    "b-",
    linewidth=2,
    marker="o",
    markersize=4,
    label="Training loss",
)
ax1.plot(
    range(1, EPOCHS + 1),
    val_losses,
    "r-",
    linewidth=2,
    marker="s",
    markersize=4,
    label="Validation loss",
)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("MSE loss (normalized scale)", fontsize=12)
ax1.set_title("LSTM training loss curves", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(OUTPUT_DIR / "loss_curve.png", dpi=180)
plt.close(fig1)

# Figure 2: four evenly spaced test windows to avoid showing only adjacent overlapping windows
fig2, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
axes = axes.flatten()
time_idx = np.arange(HORIZON) * 0.25
plot_indices = np.linspace(0, len(true_denorm) - 1, 4, dtype=int)

for i in range(4):
    ax = axes[i]
    idx = plot_indices[i]
    ax.plot(time_idx, true_denorm[idx], "b-", linewidth=2, label="Actual", alpha=0.8)
    ax.plot(
        time_idx, preds_denorm[idx], "r--", linewidth=2, label="Predicted", alpha=0.8
    )
    ax.set_xlabel("Hours after forecast origin", fontsize=12)
    ax.set_ylabel("Load (source data unit)", fontsize=12)
    ax.set_title(f"Load test sample {idx + 1}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

residuals = y_pred_flat - y_true_flat
fig2.savefig(OUTPUT_DIR / "forecast_comparison.png", dpi=180)
plt.close(fig2)
print(f"Saved best model and two figures: {OUTPUT_DIR}")

# ===================== Results summary =====================
print("\n" + "=" * 50)
print("  Results summary")
print("=" * 50)
print(f"  Test samples: {len(y_test)}")
print(f"  RMSE: {rmse:.2f} kW")
print(f"  SMAPE: {smape:.2f} p.u.")
print(f"  Mean absolute error: {np.mean(np.abs(residuals)):.2f} kW")
print("=" * 50)
