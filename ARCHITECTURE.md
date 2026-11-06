# Architecture

EMS is an offline energy management simulator for photovoltaic (PV) generation and battery storage. It forecasts the next day's PV output and demand, schedules 96 fifteen-minute intervals, and updates dispatch as observations become available.

## Core Directory Responsibilities

| Directory / File | Responsibility |
| --- | --- |
| `backend/` | API, simulation workflow, dispatch optimization, and persistence. `main.py` handles requests; `schemas.py` validates data; `service.py` advances simulations; `data.py` prepares scenarios; `optimizer.py` solves dispatch; `evaluation.py` evaluates forecasts; `repository.py`, `database.py`, and `telemetry.py` handle storage and time-series mapping. |
| `frontend/` | Streamlit interface for parameters, plan creation and execution, charts, JSON exports, and Grafana links. Calls the backend over HTTP. |
| `forecasting/` | Training and inference for the PV LSTM-Transformer and demand LSTM. `predict.py` loads and caches models without training or database access during inference. |
| `data/`, `outputs/` | PV, demand, and tariff CSV files; model weights and training artifacts, respectively. Git ignores `outputs/`. |
| `deploy/`, `scripts/` | Container dependencies, Grafana data sources and dashboards, Docker/WSL startup scripts, and stack verification. |
| `tests/` | Unit and regression tests for forecasting, dispatch, API and UI behavior, persistence, and recovery. |
| `docs/`, `runtime/` | Detailed usage, deployment, and architecture documentation; local runtime logs and verification artifacts, respectively. |
| `compose.yaml`, `Dockerfile` | Service orchestration for the frontend, API, InfluxDB, and Grafana; application image with CPU inference and Gurobi. |

## Data Flow

```text
History CSVs + model weights → forecasting/predict.py → PV / demand forecasts
Tariff CSV + forecasts → backend/data.py → simulation scenario
                                              ↓
Streamlit → HTTP API → service.py → optimizer.py (Gurobi)
                          ↓
                     repository.py → InfluxDB ← Grafana
                          ↓
                     API response → Streamlit charts / JSON export
```

1. **Create a plan:** Read historical data, run forecasts, scale power to the scenario, and combine it with tariffs. Solve a full-day, day-ahead plan subject to equipment and operating constraints.
2. **Execute with rolling updates:** Reveal the current interval's observations, correct remaining forecasts using forecast residuals, and optimize the remaining intervals from the current battery state of charge (SOC). Execute only the first interval and update SOC, costs, and history. Future observations are excluded from the current optimization and API responses.
3. **Persist and display:** Write dispatch time series to `ems_dispatch` and metrics to `ems_metrics`, then publish the `ems_simulations` snapshot. Grafana queries the selected simulation and published version; the frontend reads state through the API.

Only `service.py` advances simulation state. Checks on `expected_step` and storage versions prevent duplicate execution; running simulations become paused after a restart. Write protection relies on process-local locks, so the API requires **one instance with one worker**.

## Technology Stack

| Layer | Technologies |
| --- | --- |
| Runtime and API | Python 3.13, FastAPI, Uvicorn, Pydantic |
| Interface and charts | Streamlit, Plotly; Grafana time-series dashboards backed by InfluxDB |
| Forecasting and data processing | PyTorch, NumPy, pandas; scikit-learn and Matplotlib for training workflows |
| Dispatch optimization | gurobipy / Gurobi mixed-integer linear programming (MILP), SciPy sparse matrices; requires a valid Gurobi license |
| Persistence | InfluxDB 1.x for simulation snapshots, versioned dispatch time series, and cumulative metrics |
| Deployment and verification | Docker Compose, PowerShell / WSL startup scripts, unittest; Docker uses CPU inference, while training can use a GPU |

PV and demand data come from different regions and years and are aligned by interval and scaled for research scenarios. The system currently has no field telemetry or hardware control. See [Architecture and Data Details](docs/ARCHITECTURE.md) for storage and recovery mechanisms, and [README](README.md) for startup instructions.
