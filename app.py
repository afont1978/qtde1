
import json
import time
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_runtime import EnergyRuntime

st.set_page_config(page_title="Hybrid Quantum-Classical Control Room", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        max-width: 1600px;
    }
    .hero {
        padding: 1rem 1.2rem 1rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(25,35,55,0.95), rgba(15,20,35,0.95));
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #F4F7FB;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #C7D0DD;
    }
    .metric-card {
        padding: 0.8rem 1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .live-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-left: 0.6rem;
        color: #0b1220;
        background: #8ef0c9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SCENARIO_LABELS = {
    "economic_dispatch": "Economic dispatch",
    "grid_contingency": "Grid contingency / islanding",
    "flex_market": "Flexibility market participation",
}
MODE_LABELS = {
    "economic": "Economic",
    "resilience": "Resilience",
    "market": "Market",
}
ROUTE_COLORS = {
    "CLASSICAL": "#4E79A7",
    "QUANTUM": "#9C6ADE",
    "FALLBACK_CLASSICAL": "#F28E2B",
}


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("scenario", "economic_dispatch")
    ss.setdefault("seed", 42)
    ss.setdefault("running", False)
    ss.setdefault("sleep_s", 0.35)
    ss.setdefault("batch_steps", 6)
    ss.setdefault("live_window", 36)
    ss.setdefault("rt", EnergyRuntime(scenario=ss["scenario"], seed=ss["seed"]))


def rebuild_runtime() -> None:
    ss = st.session_state
    ss["rt"] = EnergyRuntime(scenario=ss["scenario"], seed=int(ss["seed"]))
    ss["running"] = False


def get_df() -> pd.DataFrame:
    df = st.session_state["rt"].dataframe()
    return df.copy() if not df.empty else pd.DataFrame()


def latest_record(df: pd.DataFrame) -> Dict[str, Any]:
    return {} if df.empty else df.iloc[-1].to_dict()


def safe_json_loads(text: Any) -> Any:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    if isinstance(text, (dict, list)):
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def format_route(route: str) -> str:
    return {
        "CLASSICAL": "Classical",
        "QUANTUM": "Quantum",
        "FALLBACK_CLASSICAL": "Fallback → Classical",
    }.get(route, str(route))


def route_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "decision_route" not in df.columns:
        return pd.DataFrame(columns=["route", "count"])
    vc = df["decision_route"].value_counts().reset_index()
    vc.columns = ["route", "count"]
    return vc


def active_events_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "active_event" not in df.columns:
        return pd.DataFrame(columns=["event", "count"])
    vc = df["active_event"].fillna("none").value_counts().reset_index()
    vc.columns = ["event", "count"]
    return vc


def make_line_chart(df: pd.DataFrame, x: str, y_cols: list[str], title: str, y_title: str = "", key: str = "") -> go.Figure:
    fig = go.Figure()
    for col in y_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=col, line=dict(width=2)))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
        legend=dict(orientation="h"),
        xaxis_title="Step",
        yaxis_title=y_title,
        uirevision=key or title,
        transition=dict(duration=0),
    )
    return fig


def make_area_balance_chart(df: pd.DataFrame, key: str = "") -> go.Figure:
    fig = go.Figure()
    cols = [
        ("pv_used_kw", "PV used"),
        ("battery_discharge_kw", "Battery discharge"),
        ("fuel_cell_power_kw", "Fuel cell"),
        ("grid_import_kw", "Grid import"),
    ]
    for col, name in cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["step_id"], y=df[col], stackgroup="one", mode="lines", name=name))
    if "total_load_kw" in df.columns:
        fig.add_trace(go.Scatter(x=df["step_id"], y=df["total_load_kw"], mode="lines", name="Total load", line=dict(width=3, dash="dash")))
    fig.update_layout(
        title="System energy balance",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        height=340,
        xaxis_title="Step",
        yaxis_title="kW",
        legend=dict(orientation="h"),
        uirevision=key or "system_energy_balance",
        transition=dict(duration=0),
    )
    return fig


def make_storage_chart(df: pd.DataFrame, key: str = "") -> go.Figure:
    fig = go.Figure()
    if "battery_soc" in df.columns:
        fig.add_trace(go.Scatter(x=df["step_id"], y=df["battery_soc"] * 100.0, mode="lines", name="Battery SoC [%]", line=dict(width=3)))
    if "h2_fill_ratio" in df.columns:
        fig.add_trace(go.Scatter(x=df["step_id"], y=df["h2_fill_ratio"] * 100.0, mode="lines", name="H₂ fill [%]", line=dict(width=3)))
    if "resilience_margin_h" in df.columns:
        fig.add_trace(go.Scatter(x=df["step_id"], y=df["resilience_margin_h"], mode="lines", name="Resilience margin [h]", line=dict(width=2, dash="dot"), yaxis="y2"))
    fig.update_layout(
        title="Storage and reserves",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        height=340,
        xaxis_title="Step",
        yaxis=dict(title="Percent"),
        yaxis2=dict(title="Hours", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        uirevision=key or "storage_reserves",
        transition=dict(duration=0),
    )
    return fig


def make_cost_chart(df: pd.DataFrame, key: str = "") -> go.Figure:
    fig = go.Figure()
    if "operating_cost_eur_step" in df.columns:
        fig.add_trace(go.Bar(x=df["step_id"], y=df["operating_cost_eur_step"], name="Step cost [€]"))
    if "cumulative_cost_eur" in df.columns:
        fig.add_trace(go.Scatter(x=df["step_id"], y=df["cumulative_cost_eur"], mode="lines", name="Cumulative cost [€]", yaxis="y2", line=dict(width=3)))
    fig.update_layout(
        title="Operational cost",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        height=340,
        xaxis_title="Step",
        yaxis=dict(title="Step cost [€]"),
        yaxis2=dict(title="Cumulative cost [€]", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        barmode="group",
        uirevision=key or "cost_chart",
        transition=dict(duration=0),
    )
    return fig


def make_route_chart(df: pd.DataFrame, key: str = "") -> go.Figure:
    rc = route_counts(df)
    fig = px.bar(rc, x="route", y="count", color="route", color_discrete_map=ROUTE_COLORS, template="plotly_dark", title="Decision mix")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=260, showlegend=False, uirevision=key or "decision_mix", transition=dict(duration=0))
    return fig


def make_event_chart(df: pd.DataFrame, key: str = "") -> go.Figure:
    ev = active_events_counts(df)
    fig = px.bar(ev, x="event", y="count", template="plotly_dark", title="Event frequency")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=260, showlegend=False, uirevision=key or "event_frequency", transition=dict(duration=0))
    return fig


def kpi_block(label: str, value: str, delta: str = "") -> None:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label, value, delta)
        st.markdown("</div>", unsafe_allow_html=True)


def render_overview(df: pd.DataFrame, latest: Dict[str, Any], running: bool = False) -> None:
    if df.empty:
        st.info("No simulation data yet. Press Step or Start.")
        return

    ss = st.session_state
    live_df = df.tail(int(ss["live_window"])).copy()
    mode_txt = MODE_LABELS.get(str(latest.get("mode", "")), str(latest.get("mode", "")))
    route_txt = format_route(str(latest.get("decision_route", "")))
    q_share = (df["decision_route"] == "QUANTUM").mean() * 100.0 if "decision_route" in df.columns and len(df) > 0 else 0.0
    fb_rate = df["fallback_triggered"].mean() * 100.0 if "fallback_triggered" in df.columns and len(df) > 0 else 0.0
    avg_latency = float(df["exec_ms"].tail(24).mean()) if "exec_ms" in df.columns else 0.0
    mean_conf = float(df["decision_confidence"].tail(24).mean() * 100.0) if "decision_confidence" in df.columns else 0.0

    st.markdown(
        f"### Overview {('<span class=\"live-pill\">LIVE</span>' if running else '')}",
        unsafe_allow_html=True,
    )

    row1 = st.columns(6)
    with row1[0]:
        kpi_block("Mode", mode_txt)
    with row1[1]:
        kpi_block("Total demand", f"{latest.get('total_load_kw', 0.0):.1f} kW")
    with row1[2]:
        kpi_block("Renewable share", f"{latest.get('renewable_share', 0.0) * 100:.1f}%")
    with row1[3]:
        kpi_block("Battery SoC", f"{latest.get('battery_soc', 0.0) * 100:.1f}%")
    with row1[4]:
        kpi_block("H₂ reserve", f"{latest.get('h2_level_kg', 0.0):.1f} kg")
    with row1[5]:
        kpi_block("Resilience margin", f"{latest.get('resilience_margin_h', 0.0):.2f} h")

    row2 = st.columns(6)
    with row2[0]:
        kpi_block("Grid exchange", f"{latest.get('grid_import_kw', 0.0) - latest.get('grid_export_kw', 0.0):.1f} kW")
    with row2[1]:
        kpi_block("Step cost", f"{latest.get('operating_cost_eur_step', 0.0):.2f} €")
    with row2[2]:
        kpi_block("Cumulative cost", f"{latest.get('cumulative_cost_eur', 0.0):.2f} €")
    with row2[3]:
        kpi_block("Quantum share", f"{q_share:.1f}%")
    with row2[4]:
        kpi_block("Fallback rate", f"{fb_rate:.1f}%")
    with row2[5]:
        kpi_block("Avg latency", f"{avg_latency:.0f} ms", f"Conf {mean_conf:.1f}%")

    left, right = st.columns([2.2, 1.0])
    with left:
        st.plotly_chart(make_area_balance_chart(live_df, key="overview_balance"), use_container_width=True, key="plot_overview_balance")
        c_a, c_b = st.columns(2)
        with c_a:
            st.plotly_chart(make_storage_chart(live_df, key="overview_storage"), use_container_width=True, key="plot_overview_storage")
        with c_b:
            st.plotly_chart(make_cost_chart(live_df, key="overview_cost"), use_container_width=True, key="plot_overview_cost")
    with right:
        st.plotly_chart(make_route_chart(df, key="overview_routes"), use_container_width=True, key="plot_overview_routes")
        st.plotly_chart(make_event_chart(df, key="overview_events"), use_container_width=True, key="plot_overview_events")
        st.markdown("### Current decision")
        st.write(f"**Route:** {route_txt}")
        st.write(f"**Confidence:** {latest.get('decision_confidence', 0.0) * 100:.1f}%")
        st.write(f"**Latency:** {latest.get('exec_ms', 0)} ms")
        st.write(f"**Fallback:** {'Yes' if latest.get('fallback_triggered', False) else 'No'}")
        st.write(f"**Active event:** {latest.get('active_event', 'none') or 'none'}")
        reason_text = "No fallback reason."
        reasons = latest.get("fallback_reasons")
        if isinstance(reasons, list) and reasons:
            reason_text = ", ".join(reasons)
        st.markdown("### Fallback reasons")
        st.caption(reason_text)

    st.markdown("### Live operational snapshot")
    snap_cols = [
        "step_id", "mode", "scenario", "active_event", "pv_available_kw", "pv_used_kw",
        "battery_soc", "electrolyzer_power_kw", "h2_level_kg", "fuel_cell_power_kw",
        "grid_import_kw", "grid_export_kw", "operating_cost_eur_step", "decision_route",
    ]
    snap_cols = [c for c in snap_cols if c in df.columns]
    st.dataframe(live_df[snap_cols].tail(12), use_container_width=True, height=320)


init_state()
ss = st.session_state

with st.sidebar:
    st.markdown("## Control Panel")
    selected_scenario = st.selectbox("Scenario", options=list(SCENARIO_LABELS.keys()), format_func=lambda x: SCENARIO_LABELS[x], index=list(SCENARIO_LABELS.keys()).index(ss["scenario"]))
    if selected_scenario != ss["scenario"]:
        ss["scenario"] = selected_scenario
        rebuild_runtime()
        st.rerun()

    seed = st.number_input("Seed", min_value=1, max_value=999999, value=int(ss["seed"]), step=1)
    if int(seed) != int(ss["seed"]):
        ss["seed"] = int(seed)
        rebuild_runtime()
        st.rerun()

    st.divider()
    ss["live_window"] = st.slider("Visible live window (steps)", 12, 96, int(ss["live_window"]), step=6)
    ss["batch_steps"] = st.slider("Steps per visible run", 1, 24, int(ss["batch_steps"]), step=1)
    ss["sleep_s"] = st.slider("Delay between visible steps (s)", 0.05, 1.00, float(ss["sleep_s"]), step=0.05)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Start", use_container_width=True):
            ss["running"] = True
            st.rerun()
    with c2:
        if st.button("⏸ Pause", use_container_width=True):
            ss["running"] = False
            st.rerun()

    c3, c4 = st.columns(2)
    with c3:
        if st.button("⏭ Step", use_container_width=True):
            ss["rt"].step()
            ss["running"] = False
            st.rerun()
    with c4:
        if st.button("⏹ Reset", use_container_width=True):
            rebuild_runtime()
            st.rerun()

    st.divider()
    st.markdown("### Current configuration")
    st.write(f"**Scenario:** {SCENARIO_LABELS[ss['scenario']]}")
    st.write(f"**Seed:** {ss['seed']}")
    st.write(f"**Live window:** {ss['live_window']}")
    st.write(f"**Status:** {'Running' if ss['running'] else 'Paused'}")

# Header
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Hybrid Quantum-Classical Control Room</div>
        <div class="hero-subtitle">
            Hydrogen-enabled microgrid operation with PV, battery, electrolyzer, H₂ tank, fuel cell and grid interaction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_twin, tab_audit, tab_arch = st.tabs(["Overview", "Twin drill-down", "Audit inspector", "Architecture & Use Case"])

with tab_overview:
    overview_placeholder = st.empty()
    initial_df = get_df()
    initial_latest = latest_record(initial_df)
    with overview_placeholder.container():
        render_overview(initial_df, initial_latest, running=ss["running"])

    if ss["running"]:
        # Animate visible steps inside the same run so charts visibly evolve.
        for _ in range(int(ss["batch_steps"])):
            ss["rt"].step()
            step_df = get_df()
            step_latest = latest_record(step_df)
            with overview_placeholder.container():
                render_overview(step_df, step_latest, running=True)
            time.sleep(float(ss["sleep_s"]))
        st.rerun()

with tab_twin:
    df = get_df()
    snapshots = ss["rt"].twin_snapshot()
    twin_options = ["battery", "electrolyzer", "h2_tank", "fuel_cell", "pv_array", "grid_connection", "critical_load_block", "flex_load_block"]
    twin_sel = st.selectbox("Select asset", twin_options, index=0)

    if df.empty:
        st.info("No simulation data available yet.")
    else:
        st.markdown(f"## {twin_sel.replace('_', ' ').title()}")
        if twin_sel == "battery":
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(make_line_chart(df, "step_id", ["battery_soc"], "Battery SoC", y_title="SoC", key="battery_soc_chart"), use_container_width=True, key="plot_battery_soc")
            with c2:
                st.plotly_chart(make_line_chart(df, "step_id", ["battery_charge_kw", "battery_discharge_kw"], "Battery power", y_title="kW", key="battery_power_chart"), use_container_width=True, key="plot_battery_power")
        elif twin_sel == "electrolyzer":
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(make_line_chart(df, "step_id", ["electrolyzer_power_kw"], "Electrolyzer power", y_title="kW", key="electrolyzer_power_chart"), use_container_width=True, key="plot_electrolyzer_power")
            with c2:
                st.plotly_chart(make_line_chart(df, "step_id", ["h2_production_kgph"], "H₂ production", y_title="kg/h", key="electrolyzer_h2_chart"), use_container_width=True, key="plot_electrolyzer_h2")
        elif twin_sel == "h2_tank":
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(make_line_chart(df, "step_id", ["h2_level_kg"], "H₂ tank level", y_title="kg", key="h2_level_chart"), use_container_width=True, key="plot_h2_level")
            with c2:
                st.plotly_chart(make_line_chart(df, "step_id", ["h2_fill_ratio", "resilience_margin_h"], "Reserve indicators", y_title="ratio / h", key="h2_reserve_chart"), use_container_width=True, key="plot_h2_reserve")
        elif twin_sel == "fuel_cell":
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(make_line_chart(df, "step_id", ["fuel_cell_power_kw"], "Fuel cell output", y_title="kW", key="fuel_cell_power_chart"), use_container_width=True, key="plot_fc_power")
            with c2:
                st.plotly_chart(make_line_chart(df, "step_id", ["fuel_cell_h2_use_kgph"], "Fuel cell H₂ use", y_title="kg/h", key="fuel_cell_h2_chart"), use_container_width=True, key="plot_fc_h2")
        elif twin_sel == "pv_array":
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(make_line_chart(df, "step_id", ["pv_available_kw", "pv_used_kw"], "PV availability vs dispatch", y_title="kW", key="pv_main_chart"), use_container_width=True, key="plot_pv_main")
            with c2:
                st.plotly_chart(make_line_chart(df, "step_id", ["pv_curtailed_kw"], "PV curtailment", y_title="kW", key="pv_curtailment_chart"), use_container_width=True, key="plot_pv_curtail")
        elif twin_sel == "grid_connection":
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(make_line_chart(df, "step_id", ["grid_import_kw", "grid_export_kw"], "Grid exchange", y_title="kW", key="grid_exchange_chart"), use_container_width=True, key="plot_grid_exchange")
            with c2:
                st.plotly_chart(make_line_chart(df, "step_id", ["import_price_eur_kwh", "export_price_eur_kwh"], "Grid prices", y_title="€/kWh", key="grid_price_chart"), use_container_width=True, key="plot_grid_price")
        elif twin_sel == "critical_load_block":
            st.plotly_chart(make_line_chart(df, "step_id", ["critical_load_kw", "unserved_critical_kw"], "Critical load service", y_title="kW", key="critical_load_chart"), use_container_width=True, key="plot_critical_load")
        elif twin_sel == "flex_load_block":
            st.plotly_chart(make_line_chart(df, "step_id", ["flexible_load_kw", "unserved_flex_kw"], "Flexible load service", y_title="kW", key="plot_flex_load_chart"), use_container_width=True, key="plot_flex_load")
        st.markdown("### Current twin snapshot")
        st.json(snapshots.get(twin_sel, {}))

with tab_audit:
    df = get_df()
    if df.empty:
        st.info("No records yet.")
    else:
        cols_to_show = ["step_id", "ts", "mode", "scenario", "active_event", "decision_route", "exec_ms", "decision_confidence", "operating_cost_eur_step", "cumulative_cost_eur", "resilience_margin_h", "fallback_triggered"]
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        st.dataframe(df[cols_to_show].tail(50), use_container_width=True, height=320)
        idx = st.number_input("Record index (0-based)", min_value=0, max_value=max(0, len(df) - 1), value=max(0, len(df) - 1), step=1)
        row = df.iloc[int(idx)]
        st.markdown("### Selected record")
        c1, c2 = st.columns(2)
        with c1:
            st.json({
                "step_id": int(row["step_id"]),
                "mode": row["mode"],
                "scenario": row["scenario"],
                "active_event": row["active_event"],
                "decision_route": row["decision_route"],
                "exec_ms": int(row["exec_ms"]),
                "confidence": float(row["decision_confidence"]),
                "fallback_triggered": bool(row["fallback_triggered"]),
                "fallback_reasons": row["fallback_reasons"],
            })
        with c2:
            st.json({
                "operating_cost_eur_step": float(row["operating_cost_eur_step"]),
                "cumulative_cost_eur": float(row["cumulative_cost_eur"]),
                "renewable_share": float(row["renewable_share"]),
                "resilience_margin_h": float(row["resilience_margin_h"]),
                "complexity_score": float(row["complexity_score"]),
                "discrete_ratio": float(row["discrete_ratio"]),
            })
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("### Dispatch")
            st.json(safe_json_loads(row.get("dispatch_json")))
            st.markdown("### Objective breakdown")
            st.json(safe_json_loads(row.get("objective_breakdown_json")))
        with b2:
            st.markdown("### Quantum Request Envelope")
            qre = safe_json_loads(row.get("qre_json"))
            st.json(qre if qre else {"info": "No QRE generated on this step."})
            st.markdown("### Quantum Result")
            result = safe_json_loads(row.get("result_json"))
            st.json(result if result else {"info": "No quantum result on this step."})
        st.markdown("### Export")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="energy_control_room_run.csv", mime="text/csv", key="dl_energy_csv")

with tab_arch:
    st.markdown("## Architecture & Use Case")
    st.markdown(
        """
        This verticalized prototype represents a **hydrogen-enabled microgrid** where the classical layer
        keeps physical consistency, safety and fallback behavior, while the hybrid route is used when
        the dispatch problem becomes more discrete and multiobjective.
        """
    )
    st.markdown("### Operational chain")
    st.markdown(
        """
        **Forecasting & asset state**  
        → **Problem builder**  
        → **Hybrid orchestrator**  
        → **Classical solver / Quantum gateway**  
        → **Validation & fallback**  
        → **Dispatch to assets**
        """
    )
    st.markdown("### Main assets")
    assets_df = pd.DataFrame(
        [
            ["PV Array", "Renewable generation source and curtailment origin."],
            ["Battery", "Short-duration storage, fast balancing and reserve margin support."],
            ["Electrolyzer", "Uses surplus electricity to produce hydrogen."],
            ["H₂ Tank", "Medium/long-duration reserve and resilience buffer."],
            ["Fuel Cell", "Converts hydrogen into dispatchable power during deficit or contingency."],
            ["Grid Connection", "Imports/exports energy and exposes market price signals."],
            ["Critical Loads", "Must-serve demand block protected in resilience mode."],
            ["Flexible Loads", "Shiftable or reducible demand block used for flexibility."],
        ],
        columns=["Asset", "Role"],
    )
    st.dataframe(assets_df, use_container_width=True, height=320)
    st.markdown("### Routing logic")
    st.markdown(
        """
        - **Classical route** is preferred in resilience mode, tight-latency situations, or low-complexity decisions.  
        - **Quantum route** is considered when the decision space becomes more discrete and combinatorial.  
        - **Fallback to classical** is triggered when latency or confidence becomes unacceptable.
        """
    )
    st.markdown("### Demo scenarios")
    st.markdown(
        """
        - **Economic dispatch:** minimize cost and curtailment while using battery and hydrogen intelligently.  
        - **Grid contingency / islanding:** preserve critical loads and strategic reserves.  
        - **Flexibility market participation:** optimize dispatch under market-driven incentives.
        """
    )
