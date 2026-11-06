"""Connection to the single InfluxDB service in the Compose network."""

import os

from influxdb import InfluxDBClient


class StorageError(RuntimeError):
    """A write was not acknowledged by the database."""


def connection_settings():
    return {
        "host": os.environ.get("EMS_INFLUX_HOST", "influxdb"),
        "port": int(os.environ.get("EMS_INFLUX_PORT", "8086")),
        "username": os.environ.get("EMS_INFLUX_USER", ""),
        "password": os.environ.get("EMS_INFLUX_PASSWORD", ""),
        "database": os.environ.get("EMS_INFLUX_DATABASE", "microgrid_ems"),
        "ssl": os.environ.get("EMS_INFLUX_SSL", "false").lower() == "true",
        "verify_ssl": True,
        "timeout": 10,
        "retries": 0,
    }


def create_client():
    """Connect to an existing database; create it if it does not exist."""
    settings = connection_settings()
    client = InfluxDBClient(**settings)
    try:
        client.create_database(settings["database"])
    except Exception:
        client.close()
        raise
    return client
