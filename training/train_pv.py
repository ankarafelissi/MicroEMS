"""
输入：甘肃光伏数据（17维气象特征，96个时间步）
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# 中文配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 超参数配置 =====================
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gansu_pv.csv"
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== 固定随机种子 =====================
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"[*] 使用设备: {DEVICE}")

# ===================== 1. 数据加载与预处理 =====================
print("\n[Step 1] 数据加载与预处理")
print("-" * 50)

# 保留原来的前500天；跳过前三列时间信息，其余列与原npy一致。
data = pd.read_csv(DATA_PATH, nrows=500 * 96).iloc[:, 3:].to_numpy(dtype=float)
print(f"  原始数据类型: {data.dtype}, 形状: {data.shape}")

# 提取17个特征并重塑为(500, 96)
features_list = []
for i in range(17):
    feature_96 = np.reshape(data[:, i], (500, 96))
    features_list.append(feature_96)

features = np.stack(features_list, axis=2)
real_power96 = np.reshape(data[:, 9], (500, 96))
target = real_power96

print(f"  特征形状: {features.shape}, 目标形状: {target.shape}")

# 划分数据集
train_features, val_test_features, train_target, val_test_target = train_test_split(
    features, target, test_size=1 - TRAIN_RATIO, random_state=SEED)
val_features, test_features, val_target, test_target = train_test_split(
    val_test_features, val_test_target, test_size=VAL_RATIO / (1 - TRAIN_RATIO), random_state=SEED)

print(f"  训练集: {train_features.shape}, 验证集: {val_features.shape}, 测试集: {test_features.shape}")

# 转为Tensor
train_features = torch.FloatTensor(train_features).to(DEVICE)
val_features = torch.FloatTensor(val_features).to(DEVICE)
test_features = torch.FloatTensor(test_features).to(DEVICE)
train_target = torch.FloatTensor(train_target).to(DEVICE)
val_target = torch.FloatTensor(val_target).to(DEVICE)
test_target = torch.FloatTensor(test_target).to(DEVICE)

# DataLoader
train_loader = DataLoader(TensorDataset(train_features, train_target), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_features, val_target), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(test_features, test_target), batch_size=BATCH_SIZE, shuffle=False)

# ===================== 2. LSTM-Transformer模型定义 =====================
print("\n[Step 2] 构建LSTM-Transformer模型")
print("-" * 50)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class LSTMTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(NUM_FEATURES, D_MODEL)
        self.pos_encoder = PositionalEncoding(D_MODEL, max_len=SEQ_LEN)
        self.lstm = nn.LSTM(D_MODEL, LSTM_HID, LSTM_LAYERS,
                            batch_first=False, dropout=DROPOUT if LSTM_LAYERS > 1 else 0)
        self.lstm_to_transformer = nn.Linear(LSTM_HID, D_MODEL) if LSTM_HID != D_MODEL else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_TRANSFORMER_LAYERS)
        self.output_layer = nn.Sequential(
            nn.Linear(D_MODEL, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, PRED_LEN)
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        batch_size = x.size(0)
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
    optimizer, mode='min', factor=0.5, patience=20)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  LSTM-Transformer参数量: {n_params:,}")
print(f"  d_model: {D_MODEL}, LSTM隐藏层: {LSTM_HID}, 注意力头: {NHEAD}, Transformer层: {NUM_TRANSFORMER_LAYERS}")

# ===================== 3. 模型训练 =====================
print("\n[Step 3] 模型训练")
print("-" * 50)

train_losses = []
val_losses = []
best_val_loss = float('inf')
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
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"  Epoch {epoch + 1:3d}/{EPOCHS} - 训练Loss: {train_loss:.6f}, 验证Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    if patience_counter >= patience:
        print(f"  早停触发！最佳验证损失: {best_val_loss:.6f}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("  已加载最佳模型权重")

# ===================== 4. 模型评估 =====================
print("\n[Step 4] 模型评估")
print("-" * 50)

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        predictions.append(outputs.cpu().numpy())
        actuals.append(batch_y.cpu().numpy())

predicted_power = np.concatenate(predictions, axis=0)
test_target_np = np.concatenate(actuals, axis=0)

# 计算指标
y_true_flat = test_target_np.flatten()
y_pred_flat = predicted_power.flatten()

mse = mean_squared_error(y_true_flat, y_pred_flat)
mae = mean_absolute_error(y_true_flat, y_pred_flat)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_flat, y_pred_flat)

print(f"  测试集 MSE:  {mse:.6f}")
print(f"  测试集 MAE:  {mae:.4f}")
print(f"  测试集 RMSE: {rmse:.4f}")
print(f"  测试集 R²:   {r2:.4f}")

# ===================== 5. 可视化（与负荷预测程序对齐） =====================
print("\n[Step 5] 生成可视化")
print("-" * 50)

# 图1：训练损失曲线
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='训练损失')
ax1.plot(range(1, len(val_losses) + 1), val_losses, 'r-', linewidth=2, marker='s', markersize=4, label='验证损失')
ax1.set_xlabel('训练轮次', fontsize=12)
ax1.set_ylabel('MSE损失', fontsize=12)
ax1.set_title('LSTM-Transformer训练损失曲线', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# 图2：4天预测对比（2×2子图，与负荷预测对齐）
fig2, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
axes = axes.flatten()
time_idx = np.arange(96)

for i in range(4):
    ax = axes[i]
    ax.plot(time_idx, test_target_np[i], 'b-', linewidth=2, label='真实值', alpha=0.8)
    ax.plot(time_idx, predicted_power[i], 'r--', linewidth=2, label='预测值', alpha=0.8)
    ax.set_xlabel('时间 (15分钟间隔)', fontsize=12)
    ax.set_ylabel('功率 (kW)', fontsize=12)
    ax.set_title(f'第 {i + 1} 天光伏功率预测', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# 图3：散点图（真实值 vs 预测值，与负荷预测对齐）
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10, c='#1f77b4')
min_val = min(y_true_flat.min(), y_pred_flat.min())
max_val = max(y_true_flat.max(), y_pred_flat.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线')
ax3.set_xlabel('真实功率 (kW)', fontsize=12)
ax3.set_ylabel('预测功率 (kW)', fontsize=12)
ax3.set_title(f'真实值 vs 预测值 (RMSE={rmse:.2f}, R²={r2:.4f})', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
plt.tight_layout()

# 图4：残差分布直方图（与负荷预测对齐）
residuals = y_pred_flat - y_true_flat
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.hist(residuals, bins=50, color='#5B9BD5', edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零误差线')
ax4.set_xlabel('预测误差 (kW)', fontsize=12)
ax4.set_ylabel('频数', fontsize=12)
ax4.set_title('预测残差分布', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

plt.show()

# ===================== 结果汇总 =====================
print("\n" + "=" * 50)
print("  结果汇总")
print("=" * 50)
print(f"  测试样本数: {len(test_target_np)} 天")
print(f"  MSE:  {mse:.6f}")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")
print(f"  平均绝对误差: {np.mean(np.abs(residuals)):.2f} kW")
print("=" * 50)
