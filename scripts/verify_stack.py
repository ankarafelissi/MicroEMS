"""Run inside the API container to check API -> InfluxDB -> Grafana parity.

Creates a validation plan, exercises step/start/pause, and leaves that plan paused.
Run: docker compose exec -T api python - < scripts/verify_stack.py
"""

import json
import math
import os
import time
from datetime import timedelta

import requests

from backend.database import connection_settings
from backend.repository import SimulationRepository
from backend.telemetry import POWER_FIELDS, simulation_start


def main():
    api_url = "http://127.0.0.1:8000"
    session = requests.Session()

    def api(method, path, **kwargs):
        response = session.request(method, api_url + path, timeout=40, **kwargs)
        response.raise_for_status()
        return response.json()

    assert api("GET", "/api/health")["storage"] == "influxdb"
    state = api("POST", "/api/simulations", json={})
    simulation_id = state["id"]
    route = f"/api/simulations/{simulation_id}"
    ahead = state["day_ahead"]
    api("POST", route + "/step", json={"expected_step": 0})
    duplicate = session.post(
        api_url + route + "/step", json={"expected_step": 0}, timeout=40
    )
    assert duplicate.status_code == 409
    api("POST", route + "/start", json={"interval_seconds": 0.2})
    deadline = time.monotonic() + 15
    while state["step"] < 3 and time.monotonic() < deadline:
        state = api("GET", route)
        time.sleep(0.1)
    state = api("POST", route + "/pause")
    assert state["status"] == "paused" and state["step"] >= 3
    assert state["day_ahead"] == ahead
    assert api("GET", route)["executions"] == state["executions"]

    repository = SimulationRepository()
    try:
        generation = repository._find(simulation_id)["generation"]
        counts = {}
        for phase, expected in (
            ("day_ahead", state["day_ahead"]["points"]),
            ("rolling", state["latest_plan"]["points"]),
            ("executed", state["executions"]),
        ):
            rows = list(
                repository.client.query(
                    'SELECT * FROM "ems_dispatch" WHERE "simulation_id" = $simulation_id '
                    'AND "generation" = $generation AND "phase" = $phase ORDER BY time',
                    bind_params={
                        "simulation_id": simulation_id,
                        "generation": generation,
                        "phase": phase,
                    },
                ).get_points()
            )
            assert len(rows) == len(expected), (phase, len(rows), len(expected))
            for row, point in zip(rows, expected):
                assert row["slot"] == point["slot"]
                for field in POWER_FIELDS:
                    assert math.isclose(row[field], point[field], abs_tol=1e-9), (
                        phase,
                        field,
                    )
            counts[phase] = len(rows)

        metrics = next(
            repository.client.query(
                'SELECT * FROM "ems_metrics" WHERE "simulation_id" = $simulation_id '
                'AND "generation" = $generation',
                bind_params={"simulation_id": simulation_id, "generation": generation},
            ).get_points()
        )
        for field in ("cost_yuan", "peak_import_kw", "imported_kwh", "exported_kwh"):
            assert math.isclose(metrics[field], state["metrics"][field], abs_tol=1e-9)
    finally:
        repository.close()

    grafana_url = os.environ.get("EMS_VERIFY_GRAFANA_URL", "http://grafana:3000")
    grafana = requests.Session()
    grafana.auth = (
        os.environ.get("EMS_GRAFANA_USER", "admin"),
        os.environ.get("EMS_GRAFANA_PASSWORD", "admin"),
    )
    response = grafana.get(
        grafana_url + "/api/datasources/uid/ems-influx/health", timeout=15
    )
    response.raise_for_status()
    assert response.json()["status"] == "OK"
    response = grafana.get(grafana_url + "/api/dashboards/uid/ems-dispatch", timeout=15)
    response.raise_for_status()
    dashboard = response.json()["dashboard"]
    start = simulation_start(state)
    end = start + timedelta(days=1)
    verified_queries = 0
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            query = target["query"].replace("${simulation_id}", simulation_id)
            query = query.replace("${generation}", generation).replace(
                "$timeFilter",
                f"time >= '{start.isoformat()}' AND time < '{end.isoformat()}'",
            )
            response = grafana.post(
                grafana_url + "/api/datasources/proxy/uid/ems-influx/query",
                data={"db": connection_settings()["database"], "q": query},
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()
            assert "error" not in result
            assert result["results"] and all(
                "error" not in r for r in result["results"]
            ), result
            assert any(r.get("series") for r in result["results"]), query
            verified_queries += 1
    print(
        json.dumps(
            {
                "simulation_id": simulation_id,
                "status": state["status"],
                "step": state["step"],
                "generation": generation,
                "phase_point_counts": counts,
                "grafana_queries_verified": verified_queries,
                "grafana_path": f"/d/ems-dispatch/ems-dispatch?var-simulation_id={simulation_id}"
                f"&from={int(start.timestamp() * 1000)}&to={int(end.timestamp() * 1000)}",
                "api_influx_grafana_values_match": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
