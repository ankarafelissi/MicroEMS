"""HTTP boundary: python -m uvicorn backend.main:app --port 8000."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from influxdb.exceptions import InfluxDBClientError, InfluxDBServerError
from requests.exceptions import RequestException

from backend.data import scenario_catalog
from backend.database import StorageError
from backend.repository import SimulationRepository
from backend.schemas import (
    Conflict,
    CreateSimulation,
    OptimizationError,
    SimulationState,
    SimulationSummary,
    StartRequest,
    StepRequest,
)
from backend.service import SimulationService

logger = logging.getLogger(__name__)


def create_app(repository=None):
    @asynccontextmanager
    async def lifespan(application):
        storage = repository if repository is not None else SimulationRepository()
        try:
            storage.repair_time_series()
            application.state.service = SimulationService(storage)
            try:
                yield
            finally:
                application.state.service.shutdown()
        finally:
            if repository is None:
                storage.close()

    application = FastAPI(
        title="EMS Energy Dispatch API", version="1.0.0", lifespan=lifespan
    )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request, error):
        # Keep useful field paths without echoing arbitrary input or exception objects.
        details = [
            {
                "loc": [
                    part if str(part).isascii() else "field" for part in item["loc"]
                ],
                "msg": item["msg"] if item["msg"].isascii() else "Invalid input",
                "type": item["type"],
            }
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    async def storage_error(_request, error):
        logger.error("InfluxDB request failed", exc_info=error)
        return JSONResponse(
            status_code=503,
            content={
                "detail": {
                    "code": "storage_unavailable",
                    "message": "Storage is unavailable. Check the database connection and server logs.",
                }
            },
        )

    for error_type in (
        InfluxDBClientError,
        InfluxDBServerError,
        RequestException,
        StorageError,
    ):
        application.add_exception_handler(error_type, storage_error)

    @application.exception_handler(OptimizationError)
    async def optimization_error(_request, error):
        return JSONResponse(
            status_code=409 if error.detail["code"] == "infeasible" else 503,
            content={"detail": error.detail},
        )

    @application.exception_handler(Conflict)
    async def state_conflict(_request, error):
        return JSONResponse(
            status_code=409,
            content={"detail": {"code": "state_conflict", "message": str(error)}},
        )

    @application.exception_handler(KeyError)
    async def not_found(_request, _error):
        return JSONResponse(
            status_code=404,
            content={"detail": {"code": "not_found", "message": "Plan not found."}},
        )

    @application.exception_handler(ValueError)
    async def invalid_scenario(_request, error):
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "invalid_input", "message": str(error)}},
        )

    @application.get("/")
    def hello():
        return {"message": "EMS backend is running", "docs": "/docs"}

    @application.get("/api/health")
    def health(request: Request):
        request.app.state.service.repository.client.ping()
        return {"status": "ok", "solver": "gurobi", "storage": "influxdb"}

    @application.get("/api/scenarios")
    def scenarios():
        return scenario_catalog()

    @application.post(
        "/api/simulations", response_model=SimulationState, status_code=201
    )
    def create_simulation(body: CreateSimulation, request: Request):
        return request.app.state.service.create(body)

    @application.get("/api/simulations", response_model=list[SimulationSummary])
    def list_simulations(request: Request):
        return request.app.state.service.repository.list_summaries()

    @application.get("/api/simulations/{simulation_id}", response_model=SimulationState)
    def get_simulation(simulation_id: str, request: Request):
        return request.app.state.service.get(simulation_id)

    @application.post(
        "/api/simulations/{simulation_id}/step", response_model=SimulationState
    )
    def step(simulation_id: str, body: StepRequest, request: Request):
        return request.app.state.service.step(simulation_id, body.expected_step)

    @application.post(
        "/api/simulations/{simulation_id}/start", response_model=SimulationState
    )
    def start(simulation_id: str, body: StartRequest, request: Request):
        return request.app.state.service.start(simulation_id, body.interval_seconds)

    @application.post(
        "/api/simulations/{simulation_id}/pause", response_model=SimulationState
    )
    def pause(simulation_id: str, request: Request):
        return request.app.state.service.pause(simulation_id)

    return application


app = create_app()
