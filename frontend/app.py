"""Streamlit presentation only; all simulation mutations go through FastAPI."""

import json
import os
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode, urlsplit

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from theme import STYLES

API = os.environ.get("EMS_API_URL", "http://127.0.0.1:8000").rstrip("/")
STATUS = {
    "ready": "Day-ahead plan ready",
    "running": "Running automatically",
    "paused": "Paused",
    "completed": "Day complete",
    "blocked": "Constraint conflict / stopped",
}
COLORS = {
    "ahead": "#697585",
    "latest": "#c2ef80",
    "actual": "#62ced4",
    "warning": "#e9b96b",
}

st.set_page_config(page_title="EMS · Energy Management", page_icon="⚡", layout="wide")
st.markdown(
    f"<style>{STYLES}</style>",
    unsafe_allow_html=True,
)


class ApiError(Exception):
    pass


def api(method, path, payload=None):
    try:
        response = requests.request(method, API + path, json=payload, timeout=40)
    except requests.RequestException as error:
        raise ApiError(
            "Cannot connect to the server. Please check that the service is running."
        ) from error
    if not response.ok:
        try:
            detail = response.json().get("detail", "Request failed")
        except ValueError:
            detail = f"Server returned HTTP {response.status_code}"
        if isinstance(detail, dict):
            message = str(detail.get("message", "Request failed"))
            reasons = detail.get("reasons", [])
            message += "\n" + "\n".join(f"• {reason}" for reason in reasons)
        elif isinstance(detail, list):
            message = "; ".join(
                f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in detail
            )
        else:
            message = str(detail)
        raise ApiError(message)
    return response.json()


def choose_simulation(simulation_id):
    st.session_state.simulation_id = simulation_id
    st.query_params["simulation"] = simulation_id


def line_chart(traces, ylabel, reserve=None, limit=None):
    figure = go.Figure()
    for name, points, field, color, dash, transform in traces:
        if not points:
            continue
        values = [transform(p) if transform else p[field] for p in points]
        figure.add_trace(
            go.Scatter(
                x=[p["slot"] / 4 for p in points],
                y=values,
                name=name,
                mode="lines",
                line={"color": color, "width": 2.5, "dash": dash},
                hovertemplate="%{x:.2f} h · %{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    if reserve is not None:
        figure.add_hline(
            y=reserve,
            line_dash="dot",
            line_color=COLORS["warning"],
            annotation_text="Reserve SOC",
        )
    if limit is not None:
        figure.add_hline(
            y=limit,
            line_dash="dot",
            line_color=COLORS["warning"],
            annotation_text="Grid import soft target",
        )
    figure.update_layout(
        height=285,
        margin={"t": 30, "b": 10, "l": 8, "r": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#a4adba", "family": "Inter, Segoe UI, sans-serif", "size": 11},
        legend={
            "orientation": "h",
            "y": 1.16,
            "x": 0,
            "font": {"size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="x unified",
        hoverlabel={"bgcolor": "#20262e", "font_color": "#e9edf2"},
        xaxis={
            "range": [0, 24],
            "tickvals": [0, 6, 12, 18, 24],
            "ticktext": ["00:00", "06:00", "12:00", "18:00", "24:00"],
            "gridcolor": "#20252c",
            "zeroline": False,
        },
        yaxis={
            "title": {"text": ylabel, "font": {"size": 11}},
            "gridcolor": "#20252c",
            "zerolinecolor": "#39424e",
        },
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def render_results(state):
    grafana_url = os.environ.get("EMS_GRAFANA_URL", "http://localhost:3000").rstrip("/")
    start = (
        datetime.fromisoformat(state["created_at"])
        .astimezone(UTC)
        .replace(hour=0, minute=0, second=0, microsecond=0)
    )
    query = urlencode(
        {
            "var-simulation_id": state["id"],
            "from": int(start.timestamp() * 1000),
            "to": int((start + timedelta(days=1)).timestamp() * 1000),
        }
    )
    st.link_button(
        "Open this plan in Grafana",
        f"{grafana_url}/d/ems-dispatch/ems-dispatch?{query}",
    )
    ahead = state["day_ahead"]["points"]
    latest = state["latest_plan"]["points"]
    actual = state["executions"]
    e, b = state["request"]["enterprise"], state["request"]["equipment"]
    metrics = state["metrics"]
    left, right = st.columns([2, 1])
    left.markdown("### Overview")
    right.markdown(
        f'<div class="run-status {state["status"]}"><i></i>'
        f"{STATUS[state['status']]}<span>{state['step']:02d} / 96</span></div>",
        unsafe_allow_html=True,
    )
    st.progress(state["step"] / 96)
    columns = st.columns(4)
    columns[0].metric("Current SOC", f"{state['soc'] * 100:.1f}%")
    columns[1].metric("Net electricity cost", f"¥ {metrics['cost_yuan']:.2f}")
    columns[2].metric("Peak grid import", f"{metrics['peak_import_kw']:.2f} kW")
    columns[3].metric("Target exceedances", f"{metrics['soft_excess_steps']:02d}")
    if state["last_error"]:
        error = state["last_error"]
        st.error(
            str(error["message"])
            + "\n\n"
            + "\n\n".join(str(reason) for reason in error.get("reasons", []))
        )
    if actual:
        p = actual[-1]
        end_slot = p["slot"] + 1
        end_time = f"{end_slot // 4:02d}:{end_slot % 4 * 15:02d}"
        st.caption(
            f"Last interval  {p['time']}—{end_time}　 ·　 PV {p['pv_kw']:.2f} kW"
            f"　 /　 Demand {p['load_kw']:.2f} kW"
            f"　 /　 Charge {p['charge_kw']:.2f} kW　 /　 Discharge {p['discharge_kw']:.2f} kW"
        )
    charts_tab, prediction_tab, requirements_tab, history_tab = st.tabs(
        [
            "Dispatch curves",
            "Forecast evaluation",
            "Operating results",
            "Records & export",
        ]
    )
    with charts_tab:
        c1, c2 = st.columns(2)
        with c1, st.container(border=True, key="chart_battery"):
            st.markdown("**Battery power**")
            st.caption("Positive: discharge · Negative: charge")
            transform = lambda p: p["discharge_kw"] - p["charge_kw"]
            line_chart(
                [
                    ("Day-ahead plan", ahead, "", COLORS["ahead"], "dash", transform),
                    ("Latest plan", latest, "", COLORS["latest"], "solid", transform),
                    (
                        "Actual execution",
                        actual,
                        "",
                        COLORS["actual"],
                        "solid",
                        transform,
                    ),
                ],
                "Power / kW",
            )
        with c2, st.container(border=True, key="chart_soc"):
            st.markdown("**Battery state of charge**")
            st.caption("End-of-interval SOC")
            transform = lambda p: p["soc"] * 100
            line_chart(
                [
                    ("Day-ahead plan", ahead, "", COLORS["ahead"], "dash", transform),
                    ("Latest plan", latest, "", COLORS["latest"], "solid", transform),
                    (
                        "Actual execution",
                        actual,
                        "",
                        COLORS["actual"],
                        "solid",
                        transform,
                    ),
                ],
                "SOC / %",
                reserve=e["reserve_soc"] * 100,
            )
        c1, c2 = st.columns(2)
        with c1, st.container(border=True, key="chart_pv"):
            st.markdown("**PV generation**")
            line_chart(
                [
                    (
                        "Day-ahead forecast",
                        state["forecast"],
                        "pv_kw",
                        COLORS["ahead"],
                        "dash",
                        None,
                    ),
                    (
                        "Latest forecast",
                        latest,
                        "pv_kw",
                        COLORS["latest"],
                        "solid",
                        None,
                    ),
                    ("Actual PV", actual, "pv_kw", COLORS["actual"], "solid", None),
                ],
                "Power / kW",
            )
        with c2, st.container(border=True, key="chart_grid"):
            st.markdown("**Grid import**")
            line_chart(
                [
                    (
                        "Day-ahead plan",
                        ahead,
                        "import_kw",
                        COLORS["ahead"],
                        "dash",
                        None,
                    ),
                    (
                        "Latest plan",
                        latest,
                        "import_kw",
                        COLORS["latest"],
                        "solid",
                        None,
                    ),
                    (
                        "Actual import",
                        actual,
                        "import_kw",
                        COLORS["actual"],
                        "solid",
                        None,
                    ),
                ],
                "Power / kW",
                limit=e["import_target_kw"],
            )
        c1, c2 = st.columns(2)
        with c1, st.container(border=True, key="chart_load"):
            st.markdown("**Electricity demand**")
            line_chart(
                [
                    (
                        "Day-ahead demand forecast",
                        state["forecast"],
                        "load_kw",
                        COLORS["ahead"],
                        "dash",
                        None,
                    ),
                    (
                        "Latest demand forecast",
                        latest,
                        "load_kw",
                        COLORS["latest"],
                        "solid",
                        None,
                    ),
                    (
                        "Actual demand",
                        actual,
                        "load_kw",
                        COLORS["actual"],
                        "solid",
                        None,
                    ),
                ],
                "Power / kW",
            )
        with c2, st.container(border=True, key="chart_price"):
            st.markdown("**Time-of-use tariffs**")
            line_chart(
                [
                    (
                        "Purchase price",
                        state["forecast"],
                        "buy_price",
                        COLORS["latest"],
                        "solid",
                        None,
                    ),
                    (
                        "Sale price",
                        state["forecast"],
                        "sell_price",
                        COLORS["actual"],
                        "solid",
                        None,
                    ),
                ],
                "CNY / kWh",
            )
    with prediction_tab:
        st.markdown("**Forecast accuracy**")
        rows = []
        for method, label in (
            ("day_ahead", "Day-ahead forecast"),
            ("rolling", "Rolling one-step forecast"),
        ):
            for source, name in (("pv", "PV"), ("load", "Demand")):
                score = state["prediction_metrics"][f"{method}_{source}"]
                rows.append(
                    {
                        "Forecast": label,
                        "Source": name,
                        "Samples": score["sample_count"],
                        "MAE (kW)": "—"
                        if score["mae_kw"] is None
                        else f"{score['mae_kw']:.4f}",
                        "RMSE (kW)": "—"
                        if score["rmse_kw"] is None
                        else f"{score['rmse_kw']:.4f}",
                    }
                )
        st.table(pd.DataFrame(rows))
        st.caption("Completed intervals only · MAE / RMSE in kW")
        evaluations = state["prediction_evaluations"]
        if evaluations:
            st.download_button(
                "Download forecast evaluation CSV",
                pd.DataFrame(evaluations).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"ems-{state['id'][:8]}-prediction-errors.csv",
                mime="text/csv",
            )
        else:
            st.info("Waiting for execution data")
    with requirements_tab:
        hard = b["grid_import_capacity_kw"]
        if e["import_hard_limit_kw"] is not None:
            hard = min(hard, e["import_hard_limit_kw"])
        st.table(
            pd.DataFrame(
                [
                    {
                        "Requirement": "Reserve SOC",
                        "Setting": f"≥ {e['reserve_soc']:.0%}",
                        "Result": "Met" if metrics["reserve_satisfied"] else "Violated",
                    },
                    {
                        "Requirement": "Grid import soft target",
                        "Setting": "Off"
                        if e["import_target_kw"] is None
                        else f"≤ {e['import_target_kw']:g} kW (soft target)",
                        "Result": f"Above target: {metrics['soft_excess_steps']} intervals",
                    },
                    {
                        "Requirement": "Effective import hard limit",
                        "Setting": f"≤ {hard:g} kW (including equipment capacity)",
                        "Result": "Met"
                        if metrics["hard_limit_satisfied"]
                        else "Violated",
                    },
                    {
                        "Requirement": "Final SOC",
                        "Setting": f"≥ {b['terminal_soc']:.0%}",
                        "Result": (
                            "Met"
                            if state["soc"] >= b["terminal_soc"] - 1e-5
                            else "Not met"
                        )
                        if state["status"] == "completed"
                        else "Checked at completion",
                    },
                    {
                        "Requirement": "Objective",
                        "Setting": "Cost first"
                        if e["objective"] == "cost"
                        else "Peak first",
                        "Result": "Meet the soft target first, then minimize cost"
                        if e["objective"] == "cost"
                        else "Meet the soft target first, then minimize peak and cost",
                    },
                ]
            )
        )
        st.caption(
            "Checks cover completed intervals · Generate a new plan after changing settings"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Cost without storage", f"¥ {metrics['baseline_cost_yuan']:.2f}")
        c2.metric("Savings vs. reference", f"¥ {metrics['savings_yuan']:.2f}")
        c3.metric("Imported energy", f"{metrics['imported_kwh']:.2f} kWh")
        st.caption(
            "Reference: no storage over the same intervals · Interim savings exclude remaining battery energy"
        )
    with history_tab:
        if state["revisions"]:
            selected = st.selectbox(
                "Historical revisions",
                range(len(state["revisions"])),
                format_func=lambda i: (
                    f"Revision {i + 1} · {state['forecast'][i]['time']}"
                ),
            )
            st.dataframe(
                pd.DataFrame(state["revisions"][selected]["plan"]["points"]),
                hide_index=True,
            )
        st.markdown("**Execution records**")
        st.dataframe(pd.DataFrame(actual), hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download actual execution CSV",
            pd.DataFrame(actual).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"ems-{state['id'][:8]}-actual.csv",
            mime="text/csv",
        )
        c2.download_button(
            "Download complete plan JSON",
            json.dumps(state, ensure_ascii=False, indent=2),
            file_name=f"ems-{state['id'][:8]}.json",
            mime="application/json",
        )


center = urlsplit(st.context.url or "http://localhost:8501/")
guide_url = "https://ankarafelissi.github.io/MicroEMS-Guide/?" + urlencode(
    {"center": f"{center.scheme}://{center.netloc}/"}
)
st.markdown(
    '<div class="masthead"><div class="wordmark"><span class="brand-symbol">'
    '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
    '<path d="M14 3 5 14h6l-1 7 9-11h-6l1-7Z" '
    'stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>'
    '</span>EMS<span class="brand-divider"></span>'
    '<span class="brand-label">ENERGY MANAGEMENT</span></div>'
    '<div class="masthead-actions"><div class="system-scope">'
    "PV<span> / </span>Storage<span> / </span>Grid</div>"
    f'<a class="guide-link" href="{guide_url}" target="_blank" '
    'rel="noopener noreferrer">User guide <span>↗</span></a></div></div>'
    '<div class="page-heading"><div><p class="eyebrow">OPERATIONS WORKSPACE</p>'
    "<h1>Energy Dispatch Center</h1></div>"
    '<span class="workspace-tag">Day-ahead / Intraday</span></div>',
    unsafe_allow_html=True,
)

try:
    summaries = api("GET", "/api/simulations")
except ApiError as error:
    st.error(str(error))
    st.stop()

if "simulation_id" not in st.session_state:
    saved_id = st.query_params.get("simulation")
    st.session_state.simulation_id = saved_id or (
        summaries[0]["id"] if summaries else None
    )

with st.sidebar:
    st.markdown(
        '<div class="sidebar-heading"><span class="eyebrow">CONFIGURATION</span>'
        "<h2>Configuration</h2></div>",
        unsafe_allow_html=True,
    )
    with st.form("enterprise_settings"):
        reserve = st.slider("Reserve SOC (%)", 0, 80, 30, 5)
        soft_enabled = st.checkbox("Enable import soft target", True)
        soft = st.number_input(
            "Grid import soft target (kW)", min_value=0.0, value=1.5, step=0.1
        )
        hard_enabled = st.checkbox("Enable import hard limit", False)
        hard = st.number_input(
            "Grid import hard limit (kW)", min_value=0.0, value=3.0, step=0.1
        )
        objective = st.selectbox(
            "Objective",
            ["cost", "peak"],
            format_func=lambda x: "Cost first" if x == "cost" else "Peak first",
        )
        with st.expander("Battery & equipment"):
            capacity = st.number_input(
                "Battery capacity (kWh)", min_value=0.1, value=5.0, step=0.5
            )
            power = st.number_input(
                "Charge / discharge limit (kW)", min_value=0.1, value=2.0, step=0.1
            )
            initial = st.slider("Initial SOC (%)", 0, 100, 50, 5)
            terminal = st.slider("Minimum final SOC (%)", 0, 100, 50, 5)
            max_soc = st.slider("Maximum SOC (%)", 10, 100, 90, 5)
            grid = st.number_input(
                "Grid import capacity (kW)", min_value=0.1, value=5.0, step=0.5
            )
            export = st.number_input(
                "Grid export capacity (kW)", min_value=0.0, value=3.0, step=0.5
            )
        submitted = st.form_submit_button(
            "Generate day-ahead plan", type="primary", width="stretch"
        )
    if submitted:
        payload = {
            "enterprise": {
                "reserve_soc": reserve / 100,
                "import_target_kw": soft if soft_enabled else None,
                "import_hard_limit_kw": hard if hard_enabled else None,
                "objective": objective,
            },
            "equipment": {
                "capacity_kwh": capacity,
                "charge_limit_kw": power,
                "discharge_limit_kw": power,
                "initial_soc": initial / 100,
                "max_soc": max_soc / 100,
                "terminal_soc": terminal / 100,
                "grid_import_capacity_kw": grid,
                "grid_export_capacity_kw": export,
            },
        }
        try:
            with st.spinner("Optimizing the 96-interval day-ahead plan…"):
                result = api("POST", "/api/simulations", payload)
            choose_simulation(result["id"])
            st.rerun()
        except ApiError as error:
            st.error(str(error))
    st.divider()
    if summaries:
        ids = [s["id"] for s in summaries]
        labels = {
            s["id"]: f"{s['id'][:8]} · {STATUS[s['status']]} · {s['step']}/96"
            for s in summaries
        }
        selected = st.selectbox(
            "Saved plans",
            ids,
            index=ids.index(st.session_state.simulation_id)
            if st.session_state.simulation_id in ids
            else 0,
            format_func=lambda x: labels[x],
        )
        if st.button("Open selected plan", width="stretch"):
            choose_simulation(selected)
            st.rerun()
    st.caption(
        "Automatic runs continue after closing this page. Click Pause to stop advancing."
    )


simulation_id = st.session_state.simulation_id
if not simulation_id:
    st.info("No dispatch plan yet")
    st.markdown(
        '<div class="empty-workspace"><div class="empty-orbit">'
        '<svg viewBox="0 0 80 80" fill="none" aria-hidden="true">'
        '<rect x="25" y="16" width="30" height="48" rx="6" stroke="currentColor" '
        'stroke-width="2"/><path d="M35 11h10M41 27l-8 14h12l-8 13" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"/></svg></div>'
        '<p class="eyebrow">READY TO OPTIMIZE</p><h2>Ready for dispatch</h2>'
        "<p>Configure the settings on the left and generate a day-ahead plan.</p>"
        '<div class="empty-specs"><span><b>24</b>hour horizon</span>'
        "<span><b>96</b>intervals</span><span><b>15</b>minute steps</span></div></div>",
        unsafe_allow_html=True,
    )
    st.stop()

try:
    control_state = api("GET", f"/api/simulations/{simulation_id}")
except ApiError as error:
    st.error(str(error))
    st.stop()

st.markdown(
    '<div class="control-heading"><span>Dispatch controls</span>'
    f'<span class="plan-id">PLAN / {control_state["id"][:8]}</span></div>',
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns([1.3, 1.2, 0.8, 1], vertical_alignment="bottom")
stopped = control_state["status"] in ("completed", "blocked")
step_clicked = c1.button(
    "Execute next 15 minutes",
    disabled=stopped or control_state["status"] == "running",
    width="stretch",
)
start_clicked = c2.button(
    "Auto run / resume",
    type="primary",
    disabled=stopped or control_state["status"] == "running",
    width="stretch",
)
pause_clicked = c3.button(
    "Pause", disabled=control_state["status"] != "running", width="stretch"
)
interval = c4.selectbox(
    "Step interval", [0.2, 1.0, 3.0], index=1, format_func=lambda x: f"{x:g} s / step"
)
try:
    if step_clicked:
        api(
            "POST",
            f"/api/simulations/{simulation_id}/step",
            {"expected_step": control_state["step"]},
        )
        st.rerun()
    elif start_clicked:
        api(
            "POST",
            f"/api/simulations/{simulation_id}/start",
            {"interval_seconds": interval},
        )
        st.rerun()
    elif pause_clicked:
        api("POST", f"/api/simulations/{simulation_id}/pause")
        st.rerun()
except ApiError as error:
    st.error(str(error))
    control_state = api("GET", f"/api/simulations/{simulation_id}")


@st.fragment(run_every=2 if control_state["status"] == "running" else None)
def dashboard():
    try:
        state = api("GET", f"/api/simulations/{simulation_id}")
    except ApiError as error:
        st.error(str(error))
        return
    if state["status"] != control_state["status"]:
        st.rerun()
    render_results(state)


dashboard()
