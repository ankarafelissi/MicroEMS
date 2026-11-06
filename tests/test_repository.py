"""InfluxDB persistence, write failures, concurrency and chunk limits."""

import copy
import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient
from influx_support import FakeInfluxClient, repository_fixture
from influxdb.exceptions import InfluxDBServerError
from influxdb.line_protocol import make_lines
from requests.exceptions import ConnectionError

from backend.database import StorageError, create_client
from backend.main import create_app
from backend.repository import SimulationRepository
from backend.schemas import Conflict


def snapshot(simulation_id="one"):
    return {
        "id": simulation_id,
        "created_at": "2026-10-18T12:00:00+00:00",
        "status": "paused",
        "step": 1,
        "soc": 0.5,
        "executions": [{"slot": 0, "soc": 0.5}],
        "version": 7,
        "note": "PV data\nIncludes a newline and Unicode: 25 °C",
    }


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.repository = repository_fixture(self)

    def test_persists_across_repository_instances_and_sorts_summaries(self):
        first, second = snapshot(), snapshot("two")
        second["created_at"] = "2026-10-18T13:00:00+00:00"
        self.repository.create(first)
        self.repository.create(second)
        restored = SimulationRepository(self.repository.client)
        self.assertEqual(restored.get(first["id"]), first)
        first["step"] = 2
        restored.save(first)
        self.assertEqual(self.repository.get(first["id"]), first)
        summaries = self.repository.list_summaries()
        self.assertEqual([item["id"] for item in summaries], ["two", "one"])
        self.assertEqual(summaries[1]["step"], 2)
        self.assertNotIn("note", summaries[1])

    def test_large_snapshot_then_smaller_payload_ignores_old_chunks(self):
        state = snapshot()
        state["history"] = random.Random(0).randbytes(150000).hex()
        with patch.object(
            self.repository.client,
            "write_points",
            wraps=self.repository.client.write_points,
        ) as write:
            self.repository.create(state)
        points = write.call_args.args[0]
        fields = points[0]["fields"]
        self.assertGreater(fields["payload_chunks"], 1)
        self.assertTrue(
            all(
                len(value.encode("utf-8")) < 65536
                for value in fields.values()
                if isinstance(value, str)
            )
        )
        self.assertEqual(len(make_lines({"points": points}).splitlines()), 1)
        self.assertEqual(self.repository.get(state["id"]), state)
        del state["history"]
        self.repository.save(state)
        self.assertEqual(self.repository.get(state["id"]), state)

    def test_duplicate_create_missing_id_and_parameterized_lookup(self):
        state = snapshot("one' OR '1'='1")
        self.repository.create(state)
        self.assertEqual(self.repository.get(state["id"]), state)
        with self.assertRaises(Conflict):
            self.repository.create(snapshot(state["id"]))
        with self.assertRaises(KeyError):
            self.repository.get("missing")

    def test_concurrent_writers_reject_stale_snapshot(self):
        state = snapshot()
        self.repository.create(state)
        writers = [SimulationRepository(self.repository.client) for _ in range(2)]
        copies = [self.repository.get(state["id"]) for _ in writers]

        def save(index):
            try:
                writers[index].save(copies[index])
                return "ok"
            except Conflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertCountEqual(list(pool.map(save, range(2))), ["ok", "conflict"])
        self.assertEqual(self.repository.get(state["id"])["version"], 1)

    def test_failed_write_and_invalid_numbers_do_not_advance_version(self):
        state = snapshot()
        self.repository.create(state)
        saved = copy.deepcopy(state)
        state["step"] += 1
        for failure in (ConnectionError("offline"), InfluxDBServerError("failed")):
            with self.subTest(error=type(failure).__name__):
                with (
                    patch.object(
                        self.repository.client, "write_points", side_effect=failure
                    ),
                    self.assertRaises(type(failure)),
                ):
                    self.repository.save(state)
                self.assertEqual(state["version"], saved["version"])
                self.assertEqual(self.repository.get(state["id"]), saved)
        with (
            patch.object(self.repository.client, "write_points", return_value=False),
            self.assertRaises(StorageError),
        ):
            self.repository.save(state)
        state["soc"] = float("nan")
        with self.assertRaises(ValueError):
            self.repository.save(state)
        self.assertEqual(self.repository.get(state["id"]), saved)


class ConnectionTests(unittest.TestCase):
    def test_environment_settings_and_initialization_preserve_existing_database(self):
        settings = {
            "EMS_INFLUX_HOST": "db.example",
            "EMS_INFLUX_PORT": "8443",
            "EMS_INFLUX_USER": "ems",
            "EMS_INFLUX_PASSWORD": "test-only",
            "EMS_INFLUX_DATABASE": "ems_demo",
            "EMS_INFLUX_SSL": "true",
        }
        with (
            patch.dict("os.environ", settings),
            patch("backend.database.InfluxDBClient") as factory,
        ):
            client = create_client()
        self.assertEqual(factory.call_args.kwargs["host"], "db.example")
        self.assertEqual(factory.call_args.kwargs["port"], 8443)
        self.assertTrue(factory.call_args.kwargs["ssl"])
        client.create_database.assert_called_once_with("ems_demo")
        client.drop_database.assert_not_called()

    def test_api_reports_storage_outage_and_health_checks_database(self):
        client = FakeInfluxClient()
        with (
            TestClient(create_app(SimulationRepository(client))) as api,
            self.assertLogs("backend.main", level="ERROR"),
        ):
            with patch.object(client, "ping", side_effect=ConnectionError("offline")):
                response = api.get("/api/health")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"]["code"], "storage_unavailable")
            with patch.object(
                client, "query", side_effect=InfluxDBServerError("failed")
            ):
                self.assertEqual(api.get("/api/simulations").status_code, 503)


if __name__ == "__main__":
    unittest.main()
