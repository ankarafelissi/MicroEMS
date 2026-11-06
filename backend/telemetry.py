"""Map a committed simulation to queryable 15-minute InfluxDB points."""

from datetime import UTC, datetime, timedelta

DISPATCH_MEASUREMENT = "ems_dispatch"
METRICS_MEASUREMENT = "ems_metrics"
POWER_FIELDS = (
    "pv_kw",
    "load_kw",
    "charge_kw",
    "discharge_kw",
    "import_kw",
    "export_kw",
    "curtail_kw",
    "soc",
    "buy_price",
    "sell_price",
    "cost_yuan",
    "target_excess_kw",
)


def simulation_start(state):
    """A synthetic UTC day, independent of playback speed and source CSV dates."""
    return (
        datetime.fromisoformat(state["created_at"])
        .astimezone(UTC)
        .replace(hour=0, minute=0, second=0, microsecond=0)
    )


def operation_mode(positive, negative):
    """1: import/charge, 2: export/discharge, 3: idle."""
    if positive > 1e-5:
        return 1
    if negative > 1e-5:
        return 2
    return 3


def telemetry_points(state, generation):
    """Each generation is a complete read model, published only after writing it."""
    start = simulation_start(state)
    tags = {"simulation_id": state["id"], "generation": generation}
    phases = {
        "day_ahead": state.get("day_ahead", {}).get("points", []),
        "rolling": state.get("latest_plan", {}).get("points", []),
        "executed": state["executions"],
    }
    points = []
    for phase, records in phases.items():
        issued_step = (
            state.get("latest_plan", {}).get("from_step", 0)
            if phase == "rolling"
            else -1
        )
        for record in records:
            slot = record["slot"]
            fields = {
                name: float(record[name]) for name in POWER_FIELDS if name in record
            }
            fields.update(slot=slot, issued_step=issued_step)
            fields["battery_kw"] = fields.get("discharge_kw", 0.0) - fields.get(
                "charge_kw", 0.0
            )
            fields["grid_kw"] = fields.get("import_kw", 0.0) - fields.get(
                "export_kw", 0.0
            )
            fields["grid_mode"] = operation_mode(
                fields.get("import_kw", 0), fields.get("export_kw", 0)
            )
            fields["battery_mode"] = operation_mode(
                fields.get("charge_kw", 0), fields.get("discharge_kw", 0)
            )
            points.append(
                {
                    "measurement": DISPATCH_MEASUREMENT,
                    "tags": {**tags, "phase": phase},
                    "time": (start + timedelta(minutes=15 * slot)).isoformat(),
                    "fields": fields,
                }
            )
    executed = state["executions"]
    latest = executed[-1] if executed else {}
    metrics = {
        "step": state["step"],
        "soc": float(state["soc"]),
        "status": state["status"],
        "version": state["version"],
        "cost_yuan": sum((float(p.get("cost_yuan", 0)) for p in executed), 0.0),
        "peak_import_kw": max(
            (float(p.get("import_kw", 0)) for p in executed), default=0.0
        ),
        "imported_kwh": sum(
            (float(p.get("import_kw", 0)) * 0.25 for p in executed), 0.0
        ),
        "exported_kwh": sum(
            (float(p.get("export_kw", 0)) * 0.25 for p in executed), 0.0
        ),
        "grid_mode": operation_mode(
            latest.get("import_kw", 0), latest.get("export_kw", 0)
        ),
        "battery_mode": operation_mode(
            latest.get("charge_kw", 0), latest.get("discharge_kw", 0)
        ),
    }
    points.append(
        {
            "measurement": METRICS_MEASUREMENT,
            "tags": tags,
            "time": (
                start + timedelta(minutes=15 * max(state["step"] - 1, 0))
            ).isoformat(),
            "fields": metrics,
        }
    )
    return points
