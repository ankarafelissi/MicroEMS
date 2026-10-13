"""
输入：CSV气象负荷时序数据（温度、湿度、降雨量、负荷4维特征）

"""

# ===================== 模块导入 =====================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
import warnings
import os
from pathlib import Path

# 中文配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ===================== 超参数配置 =====================
DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'quanzhou_load.csv'
WINDOW_SIZE = 96 * 7  # 输入：过去7天 = 672个点
HORIZON = 96  # 输出：预测未来1天 = 96个点
NUM_FEATURES = 4  # 特征维度：温度、湿度、降雨量、负荷
LSTM_HID = 128
LSTM_LAYERS = 2
DROPOUT = 0.2
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 40
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== 固定随机种子 =====================
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"[*] 使用设备: {DEVICE}")

# ===================== 1. 数据读取与预处理 =====================
print("\n[Step 1] 数据加载与预处理")
print("-" * 50)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"找不到文件: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df['times'] = pd.to_datetime(df['times'])
df = df.sort_values('times').reset_index(drop=True)
print(f"  原始数据行数: {len(df)}")

# 提取4个特征
features = pd.DataFrame()
features['temp'] = df['平均温度℃'].values
features['humidity'] = df['相对湿度(平均)'].values
features['precipitation'] = df['降雨量（mm）'].values
features['load'] = df['负荷'].values

# 填充缺失值
features = features.ffill().bfill().fillna(0)
raw_data = features.values

# 归一化（Min-Max）
data_min = raw_data.min(axis=0)
data_max = raw_data.max(axis=0)
data_range = data_max - data_min
data_range[data_range < 1e-8] = 1.0
normalized = (raw_data - data_min) / data_range

# 构建时间序列样本
X, y = [], []
total_need = WINDOW_SIZE + HORIZON
for i in range(len(normalized) - total_need + 1):
    X.append(normalized[i: i + WINDOW_SIZE])
    y.append(normalized[i + WINDOW_SIZE: i + total_need, -1])  # 只预测负荷列
X = np.array(X)
y = np.array(y)

n = len(X)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)
indices = np.random.permutation(n)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"  序列样本: {n}, 训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")


# 创建Dataset和DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# 数据诊断
batch = next(iter(train_loader))
print(f"  X形状: {batch[0].shape}, y形状: {batch[1].shape}")

# ===================== 2. LSTM模型定义 =====================
print("\n[Step 2] 构建LSTM模型")
print("-" * 50)


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=NUM_FEATURES,
            hidden_size=LSTM_HID,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0
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
print(f"  LSTM参数量: {n_params:,}")
print(f"  隐藏层: {LSTM_HID}, 层数: {LSTM_LAYERS}, 学习率: {LR}")

# ===================== 3. 模型训练 =====================
print("\n[Step 3] 模型训练")
print("-" * 50)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # 训练
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

    # 验证
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

    print(f"  Epoch {epoch + 1:2d}/{EPOCHS} - 训练Loss: {avg_train_loss:.6f}, 验证Loss: {avg_val_loss:.6f}")

print("\n  训练完成!")

# ===================== 4. 模型评估 =====================
print("\n[Step 4] 模型评估")
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

# 反归一化（仅负荷列）
load_min = data_min[-1]
load_range = data_range[-1]
preds_denorm = preds * load_range + load_min
true_denorm = true * load_range + load_min

# 计算指标
y_true_flat = true_denorm.flatten()
y_pred_flat = preds_denorm.flatten()

rmse = np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2))

numerator = np.abs(y_pred_flat - y_true_flat)
denominator = np.abs(y_pred_flat) + np.abs(y_true_flat)
denominator[denominator == 0] = 1e-8
smape = 200 * np.mean(numerator / denominator)

print(f"  测试集 RMSE: {rmse:.2f} kW")
print(f"  测试集 SMAPE: {smape:.2f} p.u.")

# ===================== 5. 可视化 =====================
print("\n[Step 5] 生成可视化")
print("-" * 50)

# 图1：训练损失曲线
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(range(1, EPOCHS + 1), train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='训练损失')
ax1.plot(range(1, EPOCHS + 1), val_losses, 'r-', linewidth=2, marker='s', markersize=4, label='验证损失')
ax1.set_xlabel('训练轮次', fontsize=12)
ax1.set_ylabel('MSE损失', fontsize=12)
ax1.set_title('LSTM训练损失曲线', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# 图2：4天预测对比（2×2子图）
fig2, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
axes = axes.flatten()
time_idx = np.arange(96)

for i in range(4):
    ax = axes[i]
    ax.plot(time_idx, true_denorm[i], 'b-', linewidth=2, label='真实值', alpha=0.8)
    ax.plot(time_idx, preds_denorm[i], 'r--', linewidth=2, label='预测值', alpha=0.8)
    ax.set_xlabel('时间 (15分钟间隔)', fontsize=12)
    ax.set_ylabel('负荷 (kW)', fontsize=12)
    ax.set_title(f'第 {i + 1} 天电力负荷预测', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# 图3：散点图（真实值 vs 预测值）
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10, c='#1f77b4')
min_val = min(y_true_flat.min(), y_pred_flat.min())
max_val = max(y_true_flat.max(), y_pred_flat.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线')
ax3.set_xlabel('真实负荷 (kW)', fontsize=12)
ax3.set_ylabel('预测负荷 (kW)', fontsize=12)
ax3.set_title(f'真实值 vs 预测值 (RMSE={rmse:.2f}, SMAPE={smape:.2f})', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
plt.tight_layout()

# 图4：残差分布直方图
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
print(f"  测试样本数: {len(y_test)}")
print(f"  RMSE: {rmse:.2f} kW")
print(f"  SMAPE: {smape:.2f} p.u.")
print(f"  平均绝对误差: {np.mean(np.abs(residuals)):.2f} kW")
print("=" * 50)
