"""InfluxDB test fixture; opt into a real server with EMS_TEST_INFLUXDB=1."""

import copy
import os
from uuid import uuid4

from influxdb import InfluxDBClient
from influxdb.resultset import ResultSet

from backend.database import connection_settings
from backend.repository import MEASUREMENT, SimulationRepository


class FakeInfluxClient:
    """Model point identity and field merging without requiring a server."""

    def __init__(self):
        self.points = {}

    def ping(self):
        return "1.x-test"

    def write_points(self, points):
        for point in points:
            key = (
                point["measurement"],
                tuple(sorted(point["tags"].items())),
                point["time"],
            )
            saved = self.points.setdefault(key, {})
            saved.update(copy.deepcopy(point["fields"]))
        return True

    def query(self, query, bind_params=None):
        rows = []
        for (measurement, tags, timestamp), fields in self.points.items():
            tags = dict(tags)
            if measurement != MEASUREMENT:
                continue
            if bind_params and tags["simulation_id"] != bind_params["simulation_id"]:
                continue
            row = {"time": timestamp, **tags, **copy.deepcopy(fields)}
            if not query.startswith("SELECT *"):
                row = {
                    key: row[key]
                    for key in ("time", "simulation_id", "created_at", "status", "step")
                }
            rows.append(row)
        columns = sorted({key for row in rows for key in row})
        return ResultSet(
            {
                "series": [
                    {
                        "name": MEASUREMENT,
                        "columns": columns,
                        "values": [[row.get(key) for key in columns] for row in rows],
                    }
                ]
            }
        )

    def close(self):
        pass


def repository_fixture(test):
    """A real test database is uniquely named and removed after each test."""
    if os.environ.get("EMS_TEST_INFLUXDB") != "1":
        return SimulationRepository(FakeInfluxClient())
    database = "ems_test_" + uuid4().hex
    client = InfluxDBClient(**{**connection_settings(), "database": database})
    test.addCleanup(client.close)
    client.create_database(database)
    test.addCleanup(client.drop_database, database)
    return SimulationRepository(client)
