"""加载训练好的模型，用历史数据预测未来24小时，不触发训练或数据库操作。"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


BASE_PATH = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMModel(nn.Module):
    """与负荷训练脚本保持一致，结构参数来自模型文件。"""

    def __init__(self, window_size, horizon, num_features, lstm_hid,
                 lstm_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            num_features, lstm_hid, lstm_layers, batch_first=True,
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
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class LSTMTransformerModel(nn.Module):
    """与光伏训练脚本保持一致，保留权重名称和位置编码布局。"""

    def __init__(self, seq_len, num_features, pred_len, d_model, lstm_hid,
                 lstm_layers, nhead, transformer_layers, dim_feedforward,
                 dropout):
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        self.lstm = nn.LSTM(
            d_model, lstm_hid, lstm_layers, batch_first=False,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.lstm_to_transformer = (
            nn.Linear(lstm_hid, d_model) if lstm_hid != d_model else nn.Identity()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
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
    """每个模型只加载一次；替换模型文件后需重启进程。"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    if checkpoint['model_type'] != model_type:
        raise ValueError(f'模型类型不匹配: {model_path}')
    model_class = LSTMModel if model_type == 'LSTMModel' else LSTMTransformerModel
    model = model_class(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    return model, checkpoint


def _predict(history, model, checkpoint, window_size):
    """按训练时的列顺序归一化，并将预测还原为原始功率单位。"""
    if len(history) < window_size:
        raise ValueError(f'至少需要{window_size}行历史数据，当前只有{len(history)}行')
    missing = set(checkpoint['feature_columns']) - set(history.columns)
    if missing:
        raise ValueError(f'缺少模型输入列: {sorted(missing)}')
    values = history[checkpoint['feature_columns']].iloc[-window_size:]
    values = values.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('输入窗口仍有缺失值或无穷值，请提供连续有效的历史数据')

    normalization = checkpoint['normalization']
    normalized = (
        (values - np.asarray(normalization['feature_min']))
        / np.asarray(normalization['feature_range'])
    )
    inputs = torch.as_tensor(normalized, dtype=torch.float32, device=DEVICE)
    with torch.inference_mode():
        prediction = model(inputs.unsqueeze(0)).cpu().numpy()[0]
    prediction = (
        prediction * normalization['target_range'] + normalization['target_min']
    )
    if prediction.shape != (96,) or not np.isfinite(prediction).all():
        raise ValueError('模型必须输出96个有效的功率预测值')
    return np.maximum(prediction, 0)


def predict_load(history, model_path=BASE_PATH / 'outputs/load/best_model.pt'):
    """使用最近7天数据预测负荷；times列为连续15分钟采样。"""
    model, checkpoint = load_model('LSTMModel', str(model_path))
    missing = set(checkpoint['feature_columns'] + ['times']) - set(history.columns)
    if missing:
        raise ValueError(f'缺少负荷输入列: {sorted(missing)}')
    history = history.copy()
    history['times'] = pd.to_datetime(history['times'], errors='raise')
    history = history.sort_values('times')
    window_size = checkpoint['model_config']['window_size']
    times = history['times'].iloc[-window_size:]
    if times.isna().any() or not times.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
        raise ValueError('负荷输入时间必须连续，间隔为15分钟且不能重复')
    columns = checkpoint['feature_columns']
    # 与训练一致：先在历史数据上前向填补，最多连续2点，再截取输入窗口。
    history[columns] = (
        history[columns].astype(float).replace([np.inf, -np.inf], np.nan)
        .ffill(limit=2)
    )
    return _predict(history, model, checkpoint, window_size)


def predict_pv(history, model_path=BASE_PATH / 'outputs/pv/best_model.pt'):
    """使用最近一天的17列特征预测光伏；行序须与训练CSV一致。"""
    model, checkpoint = load_model('LSTMTransformerModel', str(model_path))
    return _predict(history, model, checkpoint, checkpoint['model_config']['seq_len'])


if __name__ == '__main__':
    pv_history = pd.read_csv(BASE_PATH / 'data/gansu_pv.csv', nrows=500 * 96)
    load_history = pd.read_csv(BASE_PATH / 'data/quanzhou_load.csv')
    for name, prediction in (
        ('PV', predict_pv(pv_history)),
        ('Load', predict_load(load_history)),
    ):
        print(f'{name}: {len(prediction)} points, '
              f'range={prediction.min():.2f}..{prediction.max():.2f} '
              '(source data unit)')
