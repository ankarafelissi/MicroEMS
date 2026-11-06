# EMS · 光伏与储能能量管理

**从预测到调度的完整应用：预测光伏与负荷，优化全天电池计划，每 15 分钟滚动调整调度。**

[English](README.md) | 简体中文

`PyTorch` · `Gurobi MILP` · `FastAPI` · `Streamlit` · `InfluxDB` · `Grafana` · `Docker Compose`

## 我实现了什么

- **预测模型：**训练光伏 LSTM-Transformer 与负荷 LSTM，基于历史输入生成每天 96 个时段的预测。
- **优化调度：**实现日前计划与日内滚动优化，支持电费和购电峰值优化，并约束电池 SOC、功率、效率和电网容量。
- **完整应用：**开发 FastAPI 后端与 Streamlit 界面，支持设备配置、计划执行、预测与观测对比及结果导出。
- **存储与监控：**用 InfluxDB 保存仿真快照与时序数据，配置 Grafana 看板，实现重启恢复与重复执行保护。

```text
历史数据 → PyTorch 预测 → Gurobi 调度 → 仿真执行
                              ↑           │
                              └─ 滚动更新 ─┘
Streamlit ↔ FastAPI ↔ InfluxDB → Grafana
```

## 本地运行

需要 Python 3.13、Docker Compose，以及项目根目录下有效的 `gurobi.lic`。模型权重和许可证不上传 Git。

先训练两个模型，再启动服务：

```bash
python -m pip install -r requirements.txt
python -m forecasting.train_pv
python -m forecasting.train_load
docker compose up -d --build --wait
```

训练生成 `outputs/pv/best_model.pt` 和 `outputs/load/best_model.pt`。容器使用 CPU 推理；训练时自动使用可用 GPU。

- **操作界面：**[localhost:8501](http://localhost:8501)
- **Grafana：**[localhost:3000](http://localhost:3000/d/ems-dispatch/ems-dispatch)，本地默认账号 `admin / admin`；首次初始化前可通过 `EMS_GRAFANA_USER` 和 `EMS_GRAFANA_PASSWORD` 修改。
- **Windows 使用 WSL Ubuntu 内的 Docker：**将 Compose 命令替换为 `./scripts/ems.ps1`。

## 代码与验证

[预测模型](forecasting/) · [后端与优化器](backend/) · [界面](frontend/) · [部署配置](compose.yaml) · [测试](tests/)

详细说明：[架构与数据](docs/ARCHITECTURE.md) · [使用说明](docs/USAGE.md) · [部署说明](docs/DEPLOY.md)。

```bash
python -m unittest discover -s tests -v
```

测试覆盖模型推理、物理约束、API 与界面行为、存储及故障恢复。调度相关测试需要正常的 Gurobi 授权连接。

**项目范围：**离线研究仿真。光伏与负荷数据来自不同地区和年份，经时段对齐和缩放构成场景；执行过程回放历史观测，未连接现场硬件。
