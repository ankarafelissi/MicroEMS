"""Publish complete InfluxDB snapshots and versioned time-series read models."""

import base64
import hashlib
import json
import zlib
from threading import RLock
from time import sleep

from influxdb.exceptions import InfluxDBServerError
from requests.exceptions import RequestException

from backend.database import StorageError, create_client
from backend.schemas import Conflict
from backend.telemetry import simulation_start, telemetry_points

MEASUREMENT = "ems_simulations"
CHUNK_SIZE = 60000
WRITE_ATTEMPTS = 3


class SimulationRepository:
    # InfluxDB has no compare-and-swap. All writers must share this process.
    _write_lock = RLock()

    def __init__(self, client=None):
        self.client = client if client is not None else create_client()

    def _find(self, simulation_id):
        result = self.client.query(
            f'SELECT * FROM "{MEASUREMENT}" WHERE "simulation_id" = $simulation_id',
            bind_params={"simulation_id": simulation_id},
        )
        return next(result.get_points(), None)

    def _write(self, state):
        payload = json.dumps(
            state,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        generation = hashlib.sha256(payload).hexdigest()
        encoded = base64.b64encode(zlib.compress(payload)).decode("ascii")
        chunks = [
            encoded[i : i + CHUNK_SIZE] for i in range(0, len(encoded), CHUNK_SIZE)
        ]
        fields = {
            "created_at": state["created_at"],
            "version": state["version"],
            "status": state["status"],
            "step": state["step"],
            "soc": float(state["soc"]),
            "generation": generation,
            "simulation_start": simulation_start(state).isoformat(),
            "payload_chunks": len(chunks),
            **{f"payload_{i}": chunk for i, chunk in enumerate(chunks)},
        }
        # The snapshot is the publication pointer. A partial telemetry write is
        # invisible to Grafana because its generation has not been published.
        point = {
            "measurement": MEASUREMENT,
            "tags": {"simulation_id": state["id"]},
            "time": state["created_at"],
            "fields": fields,
        }
        self._write_points(telemetry_points(state, generation))
        try:
            self._write_points([point])
        except (RequestException, InfluxDBServerError, StorageError):
            # A lost response may follow a successful commit. Confirm before
            # reporting failure so retries cannot accidentally advance twice.
            published = self._find(state["id"])
            if published is None or published.get("generation") != generation:
                raise

    def _write_points(self, points):
        for attempt in range(WRITE_ATTEMPTS):
            try:
                if not self.client.write_points(points):
                    raise StorageError("Storage did not confirm the write.")
                return
            except (RequestException, InfluxDBServerError, StorageError):
                if attempt == WRITE_ATTEMPTS - 1:
                    raise
                sleep(0.1 * (attempt + 1))

    def repair_time_series(self):
        """Backfill existing snapshots once, without changing their step/version."""
        with self._write_lock:
            for summary in self.list_summaries():
                point = self._find(summary["id"])
                if not point.get("generation"):
                    self._write(self.get(summary["id"]))

    def create(self, state):
        with self._write_lock:
            if self._find(state["id"]) is not None:
                raise Conflict("This plan already exists and was not overwritten.")
            updated = {**state, "version": 0}
            self._write(updated)
            state["version"] = 0

    def get(self, simulation_id):
        with self._write_lock:
            point = self._find(simulation_id)
            if point is None:
                raise KeyError(simulation_id)
            encoded = "".join(
                point[f"payload_{i}"] for i in range(point["payload_chunks"])
            )
            return json.loads(zlib.decompress(base64.b64decode(encoded)))

    def save(self, state):
        with self._write_lock:
            current = self._find(state["id"])
            if current is None or current["version"] != state["version"]:
                raise Conflict(
                    "Another request updated this plan. Refresh and try again."
                )
            updated = {
                **state,
                "created_at": current["created_at"],
                "version": state["version"] + 1,
            }
            self._write(updated)
            state.update(version=updated["version"], created_at=updated["created_at"])

    def list_summaries(self):
        with self._write_lock:
            result = self.client.query(
                f'SELECT "simulation_id", "created_at", "status", "step" '
                f'FROM "{MEASUREMENT}"'
            )
            summaries = [
                {
                    "id": point["simulation_id"],
                    "created_at": point["created_at"],
                    "status": point["status"],
                    "step": point["step"],
                }
                for point in result.get_points()
            ]
            return sorted(summaries, key=lambda item: item["created_at"], reverse=True)

    def close(self):
        self.client.close()
