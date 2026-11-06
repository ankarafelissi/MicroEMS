"""Application workflow; only this layer advances time and battery state."""

import logging
from datetime import UTC, datetime
from threading import Event, RLock, Thread
from uuid import uuid4

from backend.data import (
    DT,
    SLOTS,
    prepare_scenario,
    rolling_forecast,
)
from backend.evaluation import forecast_accuracy
from backend.optimizer import solve_dispatch
from backend.schemas import (
    Conflict,
    CreateSimulation,
    OptimizationError,
    SimulationState,
)

logger = logging.getLogger(__name__)


def saved_request(state):
    """Restore the validated operating requirements saved with this run."""
    return CreateSimulation.model_validate(state["request"])


def public_state(state):
    """Public contracts omit all unrevealed actual measurements."""
    points = state["executions"]
    e = state["request"]["enterprise"]
    b = state["request"]["equipment"]
    cost = sum(p["cost_yuan"] for p in points)
    baseline = sum(
        DT
        * (
            max(p["load_kw"] - p["pv_kw"], 0) * p["buy_price"]
            - min(max(p["pv_kw"] - p["load_kw"], 0), b["grid_export_capacity_kw"])
            * p["sell_price"]
        )
        for p in points
    )
    hard = b["grid_import_capacity_kw"]
    if e["import_hard_limit_kw"] is not None:
        hard = min(hard, e["import_hard_limit_kw"])
    metrics = {
        "cost_yuan": cost,
        "baseline_cost_yuan": baseline,
        "savings_yuan": baseline - cost,
        "peak_import_kw": max((p["import_kw"] for p in points), default=0),
        "imported_kwh": sum(p["import_kw"] * DT for p in points),
        "exported_kwh": sum(p["export_kw"] * DT for p in points),
        "soft_excess_steps": sum(p["target_excess_kw"] > 1e-5 for p in points),
        "reserve_satisfied": all(p["soc"] >= e["reserve_soc"] - 1e-5 for p in points),
        "hard_limit_satisfied": all(p["import_kw"] <= hard + 1e-5 for p in points),
    }
    evaluations, prediction_metrics = forecast_accuracy(state)
    return SimulationState.model_validate(
        {
            **state,
            "request": saved_request(state),
            "metrics": metrics,
            "prediction_metrics": prediction_metrics,
            "prediction_evaluations": evaluations,
        }
    ).model_dump()


class SimulationService:
    def __init__(self, repository):
        self.repository = repository
        self.lock = RLock()
        self.workers = {}
        # Restart recovers history but never silently resumes execution.
        for summary in repository.list_summaries():
            if summary["status"] == "running":
                state = repository.get(summary["id"])
                state["status"] = "paused"
                repository.save(state)

    def create(self, request: CreateSimulation):
        scenario = prepare_scenario()
        plan = solve_dispatch(
            scenario["forecast"], request, request.equipment.initial_soc
        )
        state = {
            "id": uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "ready",
            "step": 0,
            "soc": request.equipment.initial_soc,
            "request": request.model_dump(),
            **scenario,
            "day_ahead": plan,
            "latest_plan": plan,
            "executions": [],
            "revisions": [],
            "last_error": None,
        }
        self.repository.create(state)
        return public_state(state)

    def get(self, simulation_id):
        with self.lock:
            state = self.repository.get(simulation_id)
            worker = self.workers.get(simulation_id)
            if state["status"] == "running" and (
                not worker or not worker[1].is_alive()
            ):
                state["status"] = "paused"
                self.repository.save(state)
            return public_state(state)

    def step(self, simulation_id, expected_step, automatic=False):
        with self.lock:
            state = self.repository.get(simulation_id)
            if state["step"] != expected_step:
                raise Conflict(
                    "This interval was already executed or the page is out of date. Refresh the page; no step was repeated."
                )
            if state["status"] in ("completed", "blocked"):
                raise Conflict(
                    "This plan has completed or stopped due to conflicting requirements. Create a new plan."
                )
            if not automatic and state["status"] == "running":
                raise Conflict(
                    "Automatic execution is running. Pause before executing a single step."
                )
            if automatic and state["status"] != "running":
                return public_state(state)
            request = saved_request(state)
            t = state["step"]
            forecast = rolling_forecast(
                state["forecast"],
                t,
                state["actual_pv_kw"][t],
                state["actual_load_kw"][t],
                state["pv_capacity_kw"],
            )
            try:
                plan = solve_dispatch(forecast, request, state["soc"])
            except OptimizationError as error:
                state["status"], state["last_error"] = "blocked", error.detail
                self.repository.save(state)
                raise
            action = dict(plan["points"][0])
            b = request.equipment
            soc = (
                state["soc"]
                + DT
                * (
                    action["charge_kw"] * b.efficiency
                    - action["discharge_kw"] / b.efficiency
                )
                / b.capacity_kwh
            )
            if abs(soc - action["soc"]) > 1e-5:
                raise OptimizationError(
                    "Execution SOC does not match the optimized SOC. This step was not executed.",
                    [],
                    "solver_failed",
                )
            action["soc"] = soc
            state["soc"], state["latest_plan"] = soc, plan
            state["executions"].append(action)
            state["revisions"].append({"step": t, "plan": plan})
            state["step"] = t + 1
            state["last_error"] = None
            if state["step"] == SLOTS:
                state["status"] = "completed"
            elif not automatic:
                state["status"] = "paused"
            self.repository.save(state)
            return public_state(state)

    def start(self, simulation_id, interval_seconds):
        with self.lock:
            state = self.repository.get(simulation_id)
            if state["status"] in ("completed", "blocked"):
                raise Conflict(
                    "Completed or blocked plans cannot resume. Create a new plan."
                )
            worker = self.workers.get(simulation_id)
            if state["status"] == "running" and worker and worker[1].is_alive():
                return public_state(state)
            state["status"] = "running"
            self.repository.save(state)
            stop = Event()
            thread = Thread(
                target=self._run,
                args=(simulation_id, interval_seconds, stop),
                daemon=True,
            )
            self.workers[simulation_id] = (stop, thread)
            thread.start()
            return public_state(state)

    def _run(self, simulation_id, interval, stop):
        while not stop.wait(interval):
            try:
                with self.lock:
                    state = self.repository.get(simulation_id)
                    if stop.is_set() or state["status"] != "running":
                        return
                    self.step(simulation_id, state["step"], automatic=True)
            except OptimizationError:
                return
            except Exception:
                logger.exception("Automatic simulation failed: %s", simulation_id)
                try:
                    with self.lock:
                        state = self.repository.get(simulation_id)
                        state["status"] = "paused"
                        state["last_error"] = {
                            "code": "execution_failed",
                            "message": "Automatic execution stopped. Check storage and resume when ready.",
                            "reasons": [],
                        }
                        self.repository.save(state)
                except Exception:
                    # If storage is still down, the next successful GET pauses
                    # the orphaned run. It never silently resumes after recovery.
                    logger.exception("Could not persist pause for %s", simulation_id)
                return

    def pause(self, simulation_id):
        worker = self.workers.get(simulation_id)
        if worker:
            worker[0].set()
        with self.lock:
            state = self.repository.get(simulation_id)
            if state["status"] in ("ready", "running"):
                state["status"] = "paused"
                self.repository.save(state)
            return public_state(state)

    def shutdown(self):
        for stop, _ in self.workers.values():
            stop.set()
        for _, thread in self.workers.values():
            thread.join(timeout=12)
        for summary in self.repository.list_summaries():
            if summary["status"] == "running":
                self.pause(summary["id"])
