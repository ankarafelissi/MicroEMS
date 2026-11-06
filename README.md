# EMS · Solar & Battery Energy Management

**A complete forecasting-to-dispatch application: predict solar generation and demand, optimize a 24-hour battery schedule, and update dispatch every 15 minutes.**

English | [简体中文](README.zh-CN.md)

`PyTorch` · `Gurobi MILP` · `FastAPI` · `Streamlit` · `InfluxDB` · `Grafana` · `Docker Compose`

## What I built

- **Forecasting:** trained an LSTM-Transformer for solar generation and an LSTM for demand, producing 96 forecasts per day from historical inputs.
- **Dispatch optimization:** implemented day-ahead planning and rolling reoptimization for electricity cost and peak import, subject to battery SOC, power, efficiency, and grid limits.
- **Application:** built a FastAPI backend and Streamlit interface to configure equipment, execute plans, compare forecasts with observations, and export results.
- **Persistence and monitoring:** stored simulation snapshots and time series in InfluxDB, provisioned Grafana dashboards, and added restart recovery and duplicate-step protection.

```text
Historical data → PyTorch forecasts → Gurobi dispatch → Simulated execution
                                          ↑                   │
                                          └── Rolling update ─┘
Streamlit ↔ FastAPI ↔ InfluxDB → Grafana
```

## Run locally

Requires Python 3.13, Docker Compose, and a valid Gurobi license saved as `gurobi.lic` in the project root. Model weights and license files are excluded from Git.

Train both models, then start the services:

```bash
python -m pip install -r requirements.txt
python -m forecasting.train_pv
python -m forecasting.train_load
docker compose up -d --build --wait
```

Training writes `outputs/pv/best_model.pt` and `outputs/load/best_model.pt`. Containers use CPU inference; training uses a GPU when available.

- **Application:** [localhost:8501](http://localhost:8501)
- **Grafana:** [localhost:3000](http://localhost:3000/d/ems-dispatch/ems-dispatch) — local default: `admin / admin`; override with `EMS_GRAFANA_USER` and `EMS_GRAFANA_PASSWORD` before first initialization.
- **Windows with Docker inside WSL Ubuntu:** use `./scripts/ems.ps1` instead of the Compose command.

## Code and verification

[Forecasting](forecasting/) · [Backend and optimizer](backend/) · [Interface](frontend/) · [Deployment](compose.yaml) · [Tests](tests/)

Documentation: [Architecture](ARCHITECTURE.md) · [Usage (中文)](docs/USAGE.md) · [Deployment (中文)](docs/DEPLOY.md).

```bash
python -m unittest discover -s tests -v
```

Tests cover model inference, physical constraints, API and UI behavior, persistence, and recovery. Dispatch tests require a working Gurobi license connection.

**Scope:** an offline research simulator. Solar and demand datasets come from different regions and years and are aligned and scaled for the scenario. Execution replays historical observations; no field hardware is connected.
