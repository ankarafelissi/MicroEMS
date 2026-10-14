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
from pathlib import Path

# 中文配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ===================== 超参数配置 =====================
DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'quanzhou_load.csv'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs' / 'load'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

df = pd.read_csv(DATA_PATH)
df['times'] = pd.to_datetime(df['times'])
df = df.sort_values('times').reset_index(drop=True)
input_columns = ['平均温度℃', '相对湿度(平均)', '降雨量（mm）', '负荷']
df[input_columns] = df[input_columns].astype(float)
print(f"  原始数据行数: {len(df)}")

# 提取4个特征
features = pd.DataFrame()
features['temp'] = df['平均温度℃'].values
features['humidity'] = df['相对湿度(平均)'].values
features['precipitation'] = df['降雨量（mm）'].values
features['load'] = df['负荷'].values

# 输入仅使用过去值，最多连续填补2点（30分钟）；目标保留原始实测值。
features = features.replace([np.inf, -np.inf], np.nan)
observed_load = features['load'].to_numpy(copy=True)
features = features.ffill(limit=2)
raw_data = features.values

# 先确定训练目标的结束边界（右端点不包含），仅拟合训练时期的数据。
target_points = len(raw_data) - WINDOW_SIZE
train_end = WINDOW_SIZE + int(target_points * TRAIN_RATIO)
val_end = train_end + int(target_points * VAL_RATIO)

# 归一化（Min-Max）：验证/测试只复用参数，不重新拟合或裁剪。
data_min = np.nanmin(raw_data[:train_end], axis=0)
data_max = np.nanmax(raw_data[:train_end], axis=0)
data_range = data_max - data_min
data_range[data_range < 1e-8] = 1.0
normalized = (raw_data - data_min) / data_range
normalized_load = (observed_load - data_min[-1]) / data_range[-1]
print("  归一化参数仅由训练时段拟合；指标和负荷图在反归一化后计算")

# 构建时间序列样本
X, y, sample_starts = [], [], []
total_need = WINDOW_SIZE + HORIZON
for i in range(len(normalized) - total_need + 1):
    history = normalized[i: i + WINDOW_SIZE]
    future_load = normalized_load[i + WINDOW_SIZE: i + total_need]
    if not np.isfinite(history).all() or not np.isfinite(future_load).all():
        continue
    X.append(history)
    y.append(future_load)
    sample_starts.append(i)
X = np.array(X)
y = np.array(y)

n = len(X)
# 按可预测时间段划分边界；跨越边界的96点目标窗口不属于任何集合。
# 跳过缺失窗口后仍使用原始行号，避免压缩时间导致划分错误。
target_start = np.array(sample_starts, dtype=int) + WINDOW_SIZE
target_end = target_start + HORIZON  # 右端点不包含

train_idx = np.flatnonzero(target_end <= train_end)
val_idx = np.flatnonzero((target_start >= train_end) & (target_end <= val_end))
test_idx = np.flatnonzero(target_start >= val_end)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"  序列样本: {n}, 训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
print(f"  排除缺失样本: {max(0, len(normalized) - total_need + 1) - n}")
print(f"  排除跨边界样本: {n - len(train_idx) - len(val_idx) - len(test_idx)}")


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
best_val_loss = float('inf')

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

    if avg_val_loss < best_val_loss:
        best_val_loss = float(avg_val_loss)
        checkpoint = {
            'model_type': 'LSTMModel',
            'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'model_config': {
                'window_size': WINDOW_SIZE, 'horizon': HORIZON, 'num_features': NUM_FEATURES,
                'lstm_hid': LSTM_HID, 'lstm_layers': LSTM_LAYERS, 'dropout': DROPOUT,
            },
            'feature_columns': input_columns,
            'normalization': {
                'feature_min': data_min.tolist(), 'feature_range': data_range.tolist(),
                'target_min': float(data_min[-1]), 'target_range': float(data_range[-1]),
            },
            'best_epoch': epoch + 1, 'best_val_loss': best_val_loss,
            'sampling_minutes': 15, 'power_unit': 'source_data_unit',
            'data_file': str(DATA_PATH),
            'split_samples': [len(y_train), len(y_val), len(y_test)],
            'training_config': {'seed': SEED, 'lr': LR, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS},
        }
        torch.save(checkpoint, OUTPUT_DIR / 'best_model.pt')

    print(f"  Epoch {epoch + 1:2d}/{EPOCHS} - 训练Loss: {avg_train_loss:.6f}, 验证Loss: {avg_val_loss:.6f}")

print("\n  训练完成!")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"  已加载第{checkpoint['best_epoch']}轮最佳权重")

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
checkpoint['test_metrics'] = {'rmse': float(rmse), 'smape_percent': float(smape)}
torch.save(checkpoint, OUTPUT_DIR / 'best_model.pt')

# ===================== 5. 保存两张关键图 =====================
print("\n[Step 5] 生成可视化")
print("-" * 50)

# 图1：训练损失曲线
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(range(1, EPOCHS + 1), train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='训练损失')
ax1.plot(range(1, EPOCHS + 1), val_losses, 'r-', linewidth=2, marker='s', markersize=4, label='验证损失')
ax1.set_xlabel('训练轮次', fontsize=12)
ax1.set_ylabel('MSE损失（归一化尺度）', fontsize=12)
ax1.set_title('LSTM训练损失曲线', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(OUTPUT_DIR / 'loss_curve.png', dpi=180)
plt.close(fig1)

# 图2：均匀选取4个测试窗口，避免只展示相邻的重叠窗口
fig2, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
axes = axes.flatten()
time_idx = np.arange(HORIZON) * 0.25
plot_indices = np.linspace(0, len(true_denorm) - 1, 4, dtype=int)

for i in range(4):
    ax = axes[i]
    idx = plot_indices[i]
    ax.plot(time_idx, true_denorm[idx], 'b-', linewidth=2, label='真实值', alpha=0.8)
    ax.plot(time_idx, preds_denorm[idx], 'r--', linewidth=2, label='预测值', alpha=0.8)
    ax.set_xlabel('预测起点后小时数', fontsize=12)
    ax.set_ylabel('负荷（原始数据单位）', fontsize=12)
    ax.set_title(f'负荷测试样本 {idx + 1}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

residuals = y_pred_flat - y_true_flat
fig2.savefig(OUTPUT_DIR / 'forecast_comparison.png', dpi=180)
plt.close(fig2)
print(f'Saved best model and two figures: {OUTPUT_DIR}')

# ===================== 结果汇总 =====================
print("\n" + "=" * 50)
print("  结果汇总")
print("=" * 50)
print(f"  测试样本数: {len(y_test)}")
print(f"  RMSE: {rmse:.2f} kW")
print(f"  SMAPE: {smape:.2f} p.u.")
print(f"  平均绝对误差: {np.mean(np.abs(residuals)):.2f} kW")
print("=" * 50)
