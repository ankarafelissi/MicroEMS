"""API contracts. Power: kW; energy: kWh; prices: yuan/kWh; SOC: 0..1."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EnterpriseSettings(InputModel):
    reserve_soc: float = Field(0.30, ge=0, lt=1)
    import_target_kw: float | None = Field(1.5, ge=0, le=100000)
    import_hard_limit_kw: float | None = Field(None, ge=0, le=100000)
    objective: Literal["cost", "peak"] = "cost"


class EquipmentSettings(InputModel):
    capacity_kwh: float = Field(5, gt=0, le=1000000)
    charge_limit_kw: float = Field(2, gt=0, le=100000)
    discharge_limit_kw: float = Field(2, gt=0, le=100000)
    efficiency: float = Field(0.95, gt=0, le=1)
    initial_soc: float = Field(0.5, ge=0, le=1)
    max_soc: float = Field(0.9, gt=0, le=1)
    terminal_soc: float = Field(0.5, ge=0, le=1)
    grid_import_capacity_kw: float = Field(5, gt=0, le=100000)
    grid_export_capacity_kw: float = Field(3, ge=0, le=100000)


class CreateSimulation(InputModel):
    enterprise: EnterpriseSettings = Field(default_factory=EnterpriseSettings)
    equipment: EquipmentSettings = Field(default_factory=EquipmentSettings)

    @model_validator(mode="after")
    def check_soc(self):
        e, b = self.enterprise, self.equipment
        if not e.reserve_soc <= b.initial_soc <= b.max_soc:
            raise ValueError("Initial SOC must be between reserve SOC and maximum SOC.")
        if not e.reserve_soc <= b.terminal_soc <= b.max_soc:
            raise ValueError(
                "Final SOC target must be between reserve SOC and maximum SOC."
            )
        return self


class StepRequest(InputModel):
    expected_step: int = Field(ge=0, le=95)


class StartRequest(InputModel):
    interval_seconds: float = Field(1, ge=0.2, le=60)


class ForecastPoint(BaseModel):
    slot: int
    time: str
    pv_kw: float
    load_kw: float
    buy_price: float
    sell_price: float


class DispatchPoint(ForecastPoint):
    charge_kw: float
    discharge_kw: float
    import_kw: float
    export_kw: float
    curtail_kw: float
    soc: float
    cost_yuan: float
    target_excess_kw: float


class Plan(BaseModel):
    from_step: int
    solver_status: str
    objective_value: float
    points: list[DispatchPoint]


class Revision(BaseModel):
    step: int
    plan: Plan


class Metrics(BaseModel):
    cost_yuan: float
    baseline_cost_yuan: float
    savings_yuan: float
    peak_import_kw: float
    imported_kwh: float
    exported_kwh: float
    soft_excess_steps: int
    reserve_satisfied: bool
    hard_limit_satisfied: bool


class ForecastErrorMetrics(BaseModel):
    sample_count: int
    mae_kw: float | None
    rmse_kw: float | None


class PredictionMetrics(BaseModel):
    day_ahead_pv: ForecastErrorMetrics
    day_ahead_load: ForecastErrorMetrics
    rolling_pv: ForecastErrorMetrics
    rolling_load: ForecastErrorMetrics


class ForecastEvaluation(BaseModel):
    slot: int
    time: str
    day_ahead_pv_kw: float
    day_ahead_load_kw: float
    rolling_pv_kw: float | None
    rolling_load_kw: float | None
    rolling_issued_after_slot: int | None
    actual_pv_kw: float
    actual_load_kw: float


class SimulationState(BaseModel):
    id: str
    created_at: str
    status: Literal["ready", "running", "paused", "completed", "blocked"]
    step: int
    soc: float
    request: CreateSimulation
    source_info: dict[str, str]
    forecast: list[ForecastPoint]
    day_ahead: Plan
    latest_plan: Plan
    executions: list[DispatchPoint]
    revisions: list[Revision]
    last_error: dict | None
    metrics: Metrics
    prediction_metrics: PredictionMetrics
    prediction_evaluations: list[ForecastEvaluation]


class SimulationSummary(BaseModel):
    id: str
    created_at: str
    status: str
    step: int


class Conflict(Exception):
    """An operation cannot be applied to the current simulation state."""


class OptimizationError(Exception):
    def __init__(self, message: str, reasons: list[str], status="infeasible"):
        super().__init__(message)
        self.detail = {"code": status, "message": message, "reasons": reasons}
