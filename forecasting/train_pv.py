"""
Input: previous-day Gansu PV and weather data (17 features, 96 time steps).
Target: 96 actual PV power values for the next day.
"""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# ===================== Hyperparameters =====================
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gansu_pv.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "pv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEQ_LEN = 96
NUM_FEATURES = 17
PRED_LEN = 96
D_MODEL = 128
LSTM_HID = 128
LSTM_LAYERS = 2
NHEAD = 8
NUM_TRANSFORMER_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 200
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
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

# Use the first 500 days of the prepared dataset.
df = pd.read_csv(DATA_PATH, nrows=500 * 96)
feature_columns = [
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
    "zenith_angle",
]
data = df[feature_columns].to_numpy(dtype=float)
print(f"  Raw data type: {data.dtype}, shape: {data.shape}")

# Extract 17 features and reshape each to (500, 96)
features_list = []
for i in range(17):
    feature_96 = np.reshape(data[:, i], (500, 96))
    features_list.append(feature_96)

features = np.stack(features_list, axis=2)
real_power96 = np.reshape(data[:, 9], (500, 96))
# Offset inputs and targets by one day to keep target-day actual power out of the inputs.
features = features[:-1]
target = real_power96[1:]

print(f"  Feature shape: {features.shape}, Target shape: {target.shape}")

# Split chronologically by target date to keep future samples out of training.
n = len(features)
n_train = int(n * TRAIN_RATIO)
val_end = n_train + int(n * VAL_RATIO)
train_features, train_target = features[:n_train], target[:n_train]
val_features, val_target = features[n_train:val_end], target[n_train:val_end]
test_features, test_target = features[val_end:], target[val_end:]

print(
    f"  Training: {train_features.shape}, Validation: {val_features.shape}, Test: {test_features.shape}"
)

# Fit feature ranges on training inputs only; use a separate power range fitted on training targets.
feature_min = train_features.min(axis=(0, 1))
feature_range = train_features.max(axis=(0, 1)) - feature_min
feature_range[feature_range < 1e-8] = 1.0
power_min = float(train_target.min())
power_range = float(train_target.max() - power_min)
if power_range < 1e-8:
    power_range = 1.0

# Validation/test values may exceed [0, 1]; keep them without refitting or clipping.
train_features = (train_features - feature_min) / feature_range
val_features = (val_features - feature_min) / feature_range
test_features = (test_features - feature_min) / feature_range
train_target = (train_target - power_min) / power_range
val_target = (val_target - power_min) / power_range
test_target = (test_target - power_min) / power_range
print(
    "  Normalization fitted on training data only; metrics and power plots use original units"
)

# Convert to tensors
train_features = torch.FloatTensor(train_features).to(DEVICE)
val_features = torch.FloatTensor(val_features).to(DEVICE)
test_features = torch.FloatTensor(test_features).to(DEVICE)
train_target = torch.FloatTensor(train_target).to(DEVICE)
val_target = torch.FloatTensor(val_target).to(DEVICE)
test_target = torch.FloatTensor(test_target).to(DEVICE)

# DataLoader
train_loader = DataLoader(
    TensorDataset(train_features, train_target), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(val_features, val_target), batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(test_features, test_target), batch_size=BATCH_SIZE, shuffle=False
)

# ===================== 2. LSTM-Transformer model definition =====================
print("\n[Step 2] Build the LSTM-Transformer model")
print("-" * 50)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LSTMTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(NUM_FEATURES, D_MODEL)
        self.pos_encoder = PositionalEncoding(D_MODEL, max_len=SEQ_LEN)
        self.lstm = nn.LSTM(
            D_MODEL,
            LSTM_HID,
            LSTM_LAYERS,
            batch_first=False,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0,
        )
        self.lstm_to_transformer = (
            nn.Linear(LSTM_HID, D_MODEL) if LSTM_HID != D_MODEL else nn.Identity()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=NUM_TRANSFORMER_LAYERS
        )
        self.output_layer = nn.Sequential(
            nn.Linear(D_MODEL, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, PRED_LEN),
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.input_projection(x)  # (N, 96, 128)
        x = x.transpose(0, 1)  # (96, N, 128)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)  # (96, N, 128)
        lstm_out = self.lstm_to_transformer(lstm_out)
        transformer_out = self.transformer_encoder(lstm_out)  # (96, N, 128)
        last_output = transformer_out[-1, :, :]  # (N, 128)
        output = self.output_layer(last_output)  # (N, 96)
        return output


model = LSTMTransformerModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=20
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  LSTM-Transformer parameter count: {n_params:,}")
print(
    f"  d_model: {D_MODEL}, LSTM hidden size: {LSTM_HID}, attention heads: {NHEAD}, Transformer layers: {NUM_TRANSFORMER_LAYERS}"
)

# ===================== 3. Model training =====================
print("\n[Step 3] Model training")
print("-" * 50)

train_losses = []
val_losses = []
best_val_loss = float("inf")
patience = 50
patience_counter = 0
best_model_state = None

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        checkpoint = {
            "model_type": "LSTMTransformerModel",
            "model_state_dict": {k: v.cpu() for k, v in best_model_state.items()},
            "model_config": {
                "seq_len": SEQ_LEN,
                "num_features": NUM_FEATURES,
                "pred_len": PRED_LEN,
                "d_model": D_MODEL,
                "lstm_hid": LSTM_HID,
                "lstm_layers": LSTM_LAYERS,
                "nhead": NHEAD,
                "transformer_layers": NUM_TRANSFORMER_LAYERS,
                "dim_feedforward": DIM_FEEDFORWARD,
                "dropout": DROPOUT,
            },
            "feature_columns": feature_columns[:NUM_FEATURES],
            "normalization": {
                "feature_min": feature_min.tolist(),
                "feature_range": feature_range.tolist(),
                "target_min": power_min,
                "target_range": power_range,
            },
            "best_epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "sampling_minutes": 15,
            "power_unit": "source_data_unit",
            "data_file": str(DATA_PATH),
            "split_samples": [len(train_target), len(val_target), len(test_target)],
            "training_config": {
                "seed": SEED,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            },
        }
        torch.save(checkpoint, OUTPUT_DIR / "best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"  Epoch {epoch + 1:3d}/{EPOCHS} - training loss: {train_loss:.6f}, validation loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    if patience_counter >= patience:
        print(f"  Early stopping triggered! Best validation loss: {best_val_loss:.6f}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("  Loaded best model weights")

# ===================== 4. Model evaluation =====================
print("\n[Step 4] Model evaluation")
print("-" * 50)

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        predictions.append(outputs.cpu().numpy())
        actuals.append(batch_y.cpu().numpy())

# Restore original power units before computing metrics and plotting.
predicted_power = np.concatenate(predictions, axis=0) * power_range + power_min
test_target_np = np.concatenate(actuals, axis=0) * power_range + power_min

# Compute metrics
y_true_flat = test_target_np.flatten()
y_pred_flat = predicted_power.flatten()

mse = mean_squared_error(y_true_flat, y_pred_flat)
mae = mean_absolute_error(y_true_flat, y_pred_flat)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_flat, y_pred_flat)

print(f"  Test MSE:  {mse:.6f}")
print(f"  Test MAE:  {mae:.4f}")
print(f"  Test RMSE: {rmse:.4f}")
print(f"  Test R²:   {r2:.4f}")
checkpoint["test_metrics"] = {
    "mse": float(mse),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
}
torch.save(checkpoint, OUTPUT_DIR / "best_model.pt")

# ===================== 5. Save two key figures =====================
print("\n[Step 5] Generate plots")
print("-" * 50)

# Figure 1: training loss curves
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(
    range(1, len(train_losses) + 1),
    train_losses,
    "b-",
    linewidth=2,
    marker="o",
    markersize=4,
    label="Training loss",
)
ax1.plot(
    range(1, len(val_losses) + 1),
    val_losses,
    "r-",
    linewidth=2,
    marker="s",
    markersize=4,
    label="Validation loss",
)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("MSE loss (normalized scale)", fontsize=12)
ax1.set_title("LSTM-Transformer training loss curves", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(OUTPUT_DIR / "loss_curve.png", dpi=180)
plt.close(fig1)

# Figure 2: forecast comparisons for four evenly spaced test samples (2 x 2 panels)
fig2, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
axes = axes.flatten()
time_idx = np.arange(PRED_LEN) * 0.25
plot_indices = np.linspace(0, len(test_target_np) - 1, 4, dtype=int)

for i in range(4):
    ax = axes[i]
    idx = plot_indices[i]
    ax.plot(time_idx, test_target_np[idx], "b-", linewidth=2, label="Actual", alpha=0.8)
    ax.plot(
        time_idx, predicted_power[idx], "r--", linewidth=2, label="Predicted", alpha=0.8
    )
    ax.set_xlabel("Hours after forecast origin", fontsize=12)
    ax.set_ylabel("Power (source data unit)", fontsize=12)
    ax.set_title(f"PV test sample {idx + 1}", fontsize=14)
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
print(f"  Test samples: {len(test_target_np)} days")
print(f"  MSE:  {mse:.6f}")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")
print(f"  Mean absolute error: {np.mean(np.abs(residuals)):.2f} kW")
print("=" * 50)
