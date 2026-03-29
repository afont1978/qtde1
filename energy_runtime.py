from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

Mode = Literal["economic", "resilience", "market"]
ScenarioName = Literal["economic_dispatch", "grid_contingency", "flex_market"]
Route = Literal["CLASSICAL", "QUANTUM", "FALLBACK_CLASSICAL"]
AssetType = Literal[
    "pv_array",
    "battery",
    "electrolyzer",
    "h2_tank",
    "fuel_cell",
    "grid_connection",
    "critical_load_block",
    "flex_load_block",
    "site_microgrid",
]
EventType = Literal[
    "solar_surplus",
    "price_spike",
    "grid_outage",
    "brownout",
    "reserve_breach_risk",
    "flex_request",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TwinBase:
    twin_id: str
    asset_type: AssetType
    name: str
    ts: str
    enabled: bool = True
    alarms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)

    def step(self, dt_h: float, context: Dict[str, Any]) -> None:
        raise NotImplementedError

    def apply_dispatch(self, dispatch: Dict[str, Any], dt_h: float) -> None:
        raise NotImplementedError

    def get_constraints(self) -> Dict[str, Any]:
        return {}

    def get_kpis(self) -> Dict[str, Any]:
        return {}


@dataclass
class PVArrayTwin(TwinBase):
    p_nominal_kw: float = 300.0
    inverter_limit_kw: float = 250.0
    irradiance_proxy: float = 0.0
    module_temp_proxy: float = 25.0
    p_available_kw: float = 0.0
    p_dispatched_kw: float = 0.0
    curtailment_kw: float = 0.0

    def step(self, dt_h: float, context: Dict[str, Any]) -> None:
        weather = context["weather"]
        self.irradiance_proxy = float(weather["irradiance_proxy"])
        self.module_temp_proxy = float(weather["module_temp_proxy"])
        raw = self.p_nominal_kw * self.irradiance_proxy
        self.p_available_kw = float(min(raw, self.inverter_limit_kw))
        self.p_dispatched_kw = float(min(self.p_dispatched_kw, self.p_available_kw))
        self.curtailment_kw = float(max(0.0, self.p_available_kw - self.p_dispatched_kw))

    def apply_dispatch(self, dispatch: Dict[str, Any], dt_h: float) -> None:
        used = float(dispatch.get("pv_used_kw", self.p_available_kw))
        self.p_dispatched_kw = max(0.0, min(used, self.p_available_kw))
        self.curtailment_kw = max(0.0, self.p_available_kw - self.p_dispatched_kw)

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "p_available_kw": self.p_available_kw,
            "inverter_limit_kw": self.inverter_limit_kw,
        }


@dataclass
class BatteryTwin(TwinBase):
    capacity_kwh: float = 500.0
    soc: float = 0.60
    soc_min: float = 0.15
    soc_max: float = 0.95
    p_charge_max_kw: float = 180.0
    p_discharge_max_kw: float = 180.0
    eta_charge: float = 0.96
    eta_discharge: float = 0.96
    p_charge_kw: float = 0.0
    p_discharge_kw: float = 0.0
    reserve_locked_kwh: float = 40.0
    degradation_cost_eur_per_kwh: float = 0.015
    cycle_count_proxy: float = 0.0
    degradation_state: float = 0.0

    @property
    def energy_kwh(self) -> float:
        return self.soc * self.capacity_kwh

    def step(self, dt_h: float, context: Dict[str, Any]) -> None:
        delta = (self.p_charge_kw * self.eta_charge - self.p_discharge_kw / self.eta_discharge) * dt_h
        new_energy = self.energy_kwh + delta
        new_energy = max(self.soc_min * self.capacity_kwh, min(self.soc_max * self.capacity_kwh, new_energy))
        throughput = (abs(self.p_charge_kw) + abs(self.p_discharge_kw)) * dt_h
        self.cycle_count_proxy += throughput / max(self.capacity_kwh, 1e-6)
        self.degradation_state += throughput * 1e-5
        self.soc = new_energy / self.capacity_kwh

    def apply_dispatch(self, dispatch: Dict[str, Any], dt_h: float) -> None:
        charge = float(dispatch.get("battery_charge_kw", 0.0))
        discharge = float(dispatch.get("battery_discharge_kw", 0.0))
        self.p_charge_kw = max(0.0, min(charge, self.p_charge_max_kw))
        self.p_discharge_kw = max(0.0, min(discharge, self.p_discharge_max_kw))
        if self.p_charge_kw > 0.0 and self.p_discharge_kw > 0.0:
            if self.p_charge_kw >= self.p_discharge_kw:
                self.p_discharge_kw = 0.0
            else:
                self.p_charge_kw = 0.0

    def get_constraints(self) -> Dict[str, Any]:
        usable_kwh = max(0.0, self.energy_kwh - self.reserve_locked_kwh - self.soc_min * self.capacity_kwh)
        return {
            "soc": self.soc,
            "energy_kwh": self.energy_kwh,
            "usable_discharge_kwh": usable_kwh,
            "p_charge_max_kw": self.p_charge_max_kw,
            "p_discharge_max_kw": self.p_discharge_max_kw,
        }

    def get_kpis(self) -> Dict[str, Any]:
        return {
            "battery_soc_pct": self.soc * 100.0,
            "battery_net_power_kw": self.p_discharge_kw - self.p_charge_kw,
        }


@dataclass
class ElectrolyzerTwin(TwinBase):
    p_min_kw: float = 30.0
    p_max_kw: float = 160.0
    status_on: bool = False
    p_consumption_kw: float = 0.0
    h2_production_kgph: float = 0.0
    startup_penalty_eur: float = 2.5
    specific_energy_kwh_per_kg: float = 52.0
    startup_count: int = 0

    def step(self, dt_h: float, context: Dict[str, Any]) -> None:
        if self.status_on and self.p_consumption_kw >= self.p_min_kw:
            self.h2_production_kgph = self.p_consumption_kw / self.specific_energy_kwh_per_kg
        else:
            self.h2_production_kgph = 0.0
            self.p_consumption_kw = 0.0

    def apply_dispatch(self, dispatch: Dict[str, Any], dt_h: float) -> None:
        prev = self.status_on
        self.status_on = bool(dispatch.get("electrolyzer_on", False))
        power = float(dispatch.get("electrolyzer_power_kw", 0.0))
        if self.status_on:
            self.p_consumption_kw = max(self.p_min_kw, min(power, self.p_max_kw))
        else:
            self.p_consumption_kw = 0.0
        if not prev and self.status_on:
            self.startup_count += 1

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "status_on": self.status_on,
            "p_min_kw": self.p_min_kw,
            "p_max_kw": self.p_max_kw,
        }


@dataclass
class HydrogenTankTwin(TwinBase):
    capacity_kg: float = 80.0
    level_kg: float = 35.0
    min_reserve_kg: float = 10.0
    pressure_proxy: float = 0.5

    def step(self, dt_h: float, context: Dict[str, Any]) -> None:
        produced = float(context.get("h2_produced_kg", 0.0))
        consumed = float(context.get("h2_consumed_kg", 0.0))
        self.level_kg = max(0.0, min(self.capacity_kg, self.level_kg + produced - consumed))
        self.pressure_proxy = self.level_kg / max(self.capacity_kg, 1e-6)

    def apply_dispatch(self, dispatch: Dict[str, Any], dt_h: float) -> None:
        return None

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "level_kg": self.level_kg,
            "fill_ratio": self.level_kg / max(self.capacity_kg, 1e-6),
            "reserve_kg": self.min_reserve_kg,
            "usable_kg": max(0.0, self.level_kg - self.min_reserve_kg),
        }


@dataclass
class FuelCellTwin(TwinBase):
    p_min_kw: float = 20.0
    p_max_kw: float = 120.0
    status_on: bool = False
    p_output_kw: float = 0.0
    eta_electric: float = 0.52
    startup_cost_eur: float = 3.0
    startup_count: int = 0
    h2_consumption_kgph: float = 0.0
    h2_lhv_kwhkg: float = 33.33

    def step(self, dt_h: float, context: Dict[str, Any]) -> None:
        if self.status_on and self.p_output_kw >= self.p_min_kw:
            self.h2_consumption_kgph = self.p_output_kw / max(self.eta_electric * self.h2_lhv_kwhkg, 1e-6)
        else:
            self.p_output_kw = 0.0
            self.h2_consumption_kgph = 0.0

    def apply_dispatch(self, dispatch: Dict[str, Any], dt_h: float) -> None:
        prev = self.status_on
        self.status_on = bool(dispatch.get("fuel_cell_on", False))
        power = float(dispatch.get("fuel_cell_power_kw", 0.0))
        if self.status_on:
            self.p_output_kw = max(self.p_min_kw, min(power, self.p_max_kw))
        else:
            self.p_output_kw = 0.0
        if not prev and self.status_on:
            self.startup_count += 1

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "status_on": self.status_on,
            "p_min_kw": self.p_min_kw,
            "p_max_kw": self.p_max_kw,
        }


@dataclass
class GridConnectionTwin(TwinBase):
    p_import_max_kw: float = 250.0
    p_export_max_kw: float = 150.0
    grid_status: str = "ON"
    p_import_kw: float = 0.0
    p_export_kw: float = 0.0
    import_price_eur_kwh: float = 0.16
    export_price_eur_kwh: float = 0.07
    co2_intensity_proxy: float = 0.18

    def step(self, dt_h: float, context: Dict[str, Any]) -> None:
        tariff = context["tariff"]
        grid = context["grid_signal"]
        self.import_price_eur_kwh = float(tariff["import_price_eur_kwh"])
        self.export_price_eur_kwh = float(tariff["export_price_eur_kwh"])
        self.grid_status = str(grid["grid_status"])
        if self.grid_status != "ON":
            self.p_import_kw = 0.0
            self.p_export_kw = 0.0

    def apply_dispatch(self, dispatch: Dict[str, Any], dt_h: float) -> None:
        if self.grid_status != "ON":
            self.p_import_kw = 0.0
            self.p_export_kw = 0.0
            return
        p_imp = float(dispatch.get("grid_import_kw", 0.0))
        p_exp = float(dispatch.get("grid_export_kw", 0.0))
        self.p_import_kw = max(0.0, min(p_imp, self.p_import_max_kw))
        self.p_export_kw = max(0.0, min(p_exp, self.p_export_max_kw))
        if self.p_import_kw > 0.0 and self.p_export_kw > 0.0:
            self.p_export_kw = 0.0


@dataclass
class LoadBlockTwin(TwinBase):
    is_critical: bool = False
    requested_kw: float = 0.0
    served_kw: float = 0.0
    unserved_kw: float = 0.0
    shiftable_energy_kwh: float = 0.0
    discomfort_penalty_eur_per_kwh: float = 0.2

    def step(self, dt_h: float, context: Dict[str, Any]) -> None:
        load_ctx = context["loads"][self.twin_id]
        self.requested_kw = float(load_ctx["requested_kw"])

    def apply_dispatch(self, dispatch: Dict[str, Any], dt_h: float) -> None:
        key = "critical_served_kw" if self.is_critical else "flex_served_kw"
        served = float(dispatch.get(key, self.requested_kw))
        self.served_kw = min(self.requested_kw, max(0.0, served))
        self.unserved_kw = max(0.0, self.requested_kw - self.served_kw)


@dataclass
class SiteMicrogridTwin(TwinBase):
    mode: Mode = "economic"
    scenario: ScenarioName = "economic_dispatch"
    total_load_kw: float = 0.0
    critical_load_kw: float = 0.0
    flexible_load_kw: float = 0.0
    pv_available_kw: float = 0.0
    pv_used_kw: float = 0.0
    pv_curtailed_kw: float = 0.0
    battery_soc: float = 0.0
    battery_net_kw: float = 0.0
    electrolyzer_power_kw: float = 0.0
    h2_level_kg: float = 0.0
    fuel_cell_power_kw: float = 0.0
    grid_import_kw: float = 0.0
    grid_export_kw: float = 0.0
    renewable_share: float = 0.0
    resilience_margin_h: float = 0.0
    operating_cost_eur_step: float = 0.0
    cumulative_cost_eur: float = 0.0
    curtailment_kw: float = 0.0
    unserved_critical_kw: float = 0.0
    decision_route: Route = "CLASSICAL"
    decision_confidence: float = 0.0
    exec_ms: int = 0
    fallback_triggered: bool = False
    active_event: Optional[str] = None

    def step(self, dt_h: float, context: Dict[str, Any]) -> None:
        return None

    def apply_dispatch(self, dispatch: Dict[str, Any], dt_h: float) -> None:
        return None


@dataclass
class ScenarioEvent:
    event_type: EventType
    severity: float
    start_step: int
    end_step: int
    payload: Dict[str, Any]

    def is_active(self, step_id: int) -> bool:
        return self.start_step <= step_id <= self.end_step


@dataclass
class ScenarioContext:
    scenario: ScenarioName
    mode: Mode
    weather: Dict[str, Any]
    tariff: Dict[str, Any]
    grid_signal: Dict[str, Any]
    loads: Dict[str, Dict[str, Any]]
    active_events: List[ScenarioEvent] = field(default_factory=list)


@dataclass
class DispatchProblem:
    step_id: int
    mode: Mode
    scenario: ScenarioName
    objective_name: str
    constraints: Dict[str, Any]
    objective_terms: Dict[str, float]
    complexity_score: float
    discrete_ratio: float
    horizon_steps: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnergyExecRecord:
    step_id: int
    ts: str
    mode: Mode
    scenario: ScenarioName
    active_event: Optional[str]
    total_load_kw: float
    critical_load_kw: float
    flexible_load_kw: float
    pv_available_kw: float
    pv_used_kw: float
    pv_curtailed_kw: float
    battery_soc: float
    battery_charge_kw: float
    battery_discharge_kw: float
    battery_energy_kwh: float
    electrolyzer_on: bool
    electrolyzer_power_kw: float
    h2_production_kgph: float
    h2_level_kg: float
    h2_fill_ratio: float
    fuel_cell_on: bool
    fuel_cell_power_kw: float
    fuel_cell_h2_use_kgph: float
    grid_import_kw: float
    grid_export_kw: float
    import_price_eur_kwh: float
    export_price_eur_kwh: float
    renewable_share: float
    resilience_margin_h: float
    operating_cost_eur_step: float
    cumulative_cost_eur: float
    curtailment_kw: float
    unserved_critical_kw: float
    unserved_flex_kw: float
    decision_route: Route
    decision_confidence: float
    exec_ms: int
    latency_breach: bool
    fallback_triggered: bool
    fallback_reasons: List[str]
    complexity_score: float
    discrete_ratio: float
    qre_json: Optional[str] = None
    result_json: Optional[str] = None
    dispatch_json: Optional[str] = None
    objective_breakdown_json: Optional[str] = None


class ClassicalEnergySolver:
    def solve(self, state: Dict[str, Any], problem: DispatchProblem) -> Tuple[Dict[str, Any], Dict[str, float], float]:
        crit = state["critical_load_kw"]
        flex = state["flexible_load_kw"]
        demand = crit + flex
        pv = state["pv_available_kw"]
        batt_soc = state["battery_soc"]
        batt_energy = state["battery_energy_kwh"]
        batt_can_discharge = max(0.0, batt_energy - 40.0 - 0.15 * 500.0)
        batt_discharge_cap_kw = min(180.0, batt_can_discharge / (5.0 / 60.0))
        h2_level = state["h2_level_kg"]
        grid_status = state["grid_status"]
        import_price = state["import_price_eur_kwh"]
        event = state["active_event"]
        mode = problem.mode

        pv_to_load = min(pv, demand)
        residual = demand - pv_to_load

        battery_discharge_kw = 0.0
        battery_charge_kw = 0.0
        electrolyzer_on = False
        electrolyzer_power_kw = 0.0
        fuel_cell_on = False
        fuel_cell_power_kw = 0.0
        grid_import_kw = 0.0
        grid_export_kw = 0.0

        flex_served_kw = flex
        critical_served_kw = crit

        if mode == "resilience" or grid_status != "ON":
            flex_served_kw = min(flex, max(0.0, pv - crit))
            demand = crit + flex_served_kw
            pv_to_load = min(pv, demand)
            residual = demand - pv_to_load

            battery_discharge_kw = min(residual, batt_discharge_cap_kw)
            residual -= battery_discharge_kw

            if residual > 0 and h2_level > 12.0:
                fuel_cell_on = True
                fuel_cell_power_kw = min(120.0, residual)
                residual -= fuel_cell_power_kw

            if residual > 0:
                flex_served_kw = max(0.0, flex_served_kw - residual)
                residual = 0.0
        else:
            if pv > demand:
                surplus = pv - demand
                if batt_soc < 0.88:
                    battery_charge_kw = min(surplus, 180.0)
                    surplus -= battery_charge_kw
                if surplus > 25.0 and h2_level < 72.0:
                    electrolyzer_on = True
                    electrolyzer_power_kw = min(160.0, max(30.0, surplus))
                    surplus -= electrolyzer_power_kw
                if surplus > 0 and grid_status == "ON":
                    grid_export_kw = min(150.0, surplus)
            else:
                if import_price > 0.20 and batt_soc > 0.30:
                    battery_discharge_kw = min(residual, batt_discharge_cap_kw)
                    residual -= battery_discharge_kw

                if residual > 0 and event == "price_spike" and h2_level > 16.0:
                    fuel_cell_on = True
                    fuel_cell_power_kw = min(120.0, residual)
                    residual -= fuel_cell_power_kw

                if residual > 0 and grid_status == "ON":
                    grid_import_kw = min(250.0, residual)
                    residual -= grid_import_kw

                if residual > 0:
                    flex_served_kw = max(0.0, flex - residual)

        dispatch = {
            "pv_used_kw": min(pv, critical_served_kw + flex_served_kw + battery_charge_kw + electrolyzer_power_kw + grid_export_kw),
            "battery_charge_kw": battery_charge_kw,
            "battery_discharge_kw": battery_discharge_kw,
            "electrolyzer_on": electrolyzer_on,
            "electrolyzer_power_kw": electrolyzer_power_kw,
            "fuel_cell_on": fuel_cell_on,
            "fuel_cell_power_kw": fuel_cell_power_kw,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": grid_export_kw,
            "critical_served_kw": critical_served_kw,
            "flex_served_kw": flex_served_kw,
        }

        objective_breakdown = {
            "grid_import_cost": grid_import_kw * import_price * (5.0 / 60.0),
            "battery_degradation_cost": (battery_charge_kw + battery_discharge_kw) * 0.015 * (5.0 / 60.0),
            "electrolyzer_startup_cost": 0.0 if not electrolyzer_on else 0.25,
            "fuel_cell_startup_cost": 0.0 if not fuel_cell_on else 0.35,
            "curtailment_penalty": max(0.0, pv - dispatch["pv_used_kw"]) * 0.04 * (5.0 / 60.0),
            "unserved_critical_penalty": max(0.0, crit - critical_served_kw) * 5.0 * (5.0 / 60.0),
            "unserved_flex_penalty": max(0.0, flex - flex_served_kw) * 0.4 * (5.0 / 60.0),
            "export_revenue": -grid_export_kw * state["export_price_eur_kwh"] * (5.0 / 60.0),
        }
        confidence = 0.88 if mode == "resilience" else 0.80
        return dispatch, objective_breakdown, confidence


class MockQuantumEnergySolver:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def solve(self, state: Dict[str, Any], problem: DispatchProblem) -> Tuple[Dict[str, Any], Dict[str, float], float, Dict[str, Any], Dict[str, Any]]:
        classical = ClassicalEnergySolver()
        dispatch, obj, _ = classical.solve(state, problem)

        pv = state["pv_available_kw"]
        demand = state["total_load_kw"]
        h2_level = state["h2_level_kg"]
        batt_soc = state["battery_soc"]

        if problem.mode == "economic" and pv > demand and batt_soc > 0.75 and h2_level < 70.0:
            dispatch["battery_charge_kw"] = max(0.0, dispatch["battery_charge_kw"] - 20.0)
            dispatch["electrolyzer_on"] = True
            dispatch["electrolyzer_power_kw"] = min(160.0, max(30.0, dispatch["electrolyzer_power_kw"] + 20.0))

        if problem.mode == "market" and state["grid_status"] == "ON" and state["export_price_eur_kwh"] > 0.10:
            dispatch["grid_export_kw"] = min(150.0, dispatch["grid_export_kw"] + 20.0)

        if problem.mode == "economic" and state["active_event"] == "price_spike" and h2_level > 18.0:
            dispatch["fuel_cell_on"] = True
            dispatch["fuel_cell_power_kw"] = min(120.0, max(dispatch["fuel_cell_power_kw"], 40.0))
            dispatch["grid_import_kw"] = max(0.0, dispatch["grid_import_kw"] - 20.0)

        obj["grid_import_cost"] *= 0.95
        obj["curtailment_penalty"] *= 0.90
        confidence = 0.76 + 0.08 * self.rng.random()

        qre = {
            "qre_version": "1.0",
            "mode": problem.mode,
            "scenario": problem.scenario,
            "complexity_score": problem.complexity_score,
            "discrete_ratio": problem.discrete_ratio,
            "objective_name": problem.objective_name,
            "constraints": problem.constraints,
        }
        result = {
            "status": "SUCCEEDED",
            "backend": {
                "provider": "SIM_QPU",
                "backend_id": "sim-energy-qpu",
                "queue_ms": int(250 + 400 * self.rng.random()),
                "exec_ms": int(180 + 140 * self.rng.random()),
            },
            "solution": {
                "dispatch": dispatch,
                "confidence": confidence,
            },
        }
        return dispatch, obj, confidence, qre, result


class EnergyHybridOrchestrator:
    def __init__(self, seed: int = 42):
        self.classical = ClassicalEnergySolver()
        self.quantum = MockQuantumEnergySolver(seed=seed)

    def choose_route(self, problem: DispatchProblem) -> Route:
        if problem.mode == "resilience":
            return "CLASSICAL"
        if problem.complexity_score < 4.0:
            return "CLASSICAL"
        if problem.discrete_ratio >= 0.45 and problem.horizon_steps >= 8:
            return "QUANTUM"
        return "CLASSICAL"

    def solve(self, state: Dict[str, Any], problem: DispatchProblem) -> Dict[str, Any]:
        route = self.choose_route(problem)

        if route == "CLASSICAL":
            dispatch, breakdown, confidence = self.classical.solve(state, problem)
            return {
                "route": "CLASSICAL",
                "dispatch": dispatch,
                "objective_breakdown": breakdown,
                "confidence": confidence,
                "exec_ms": 55,
                "latency_breach": False,
                "fallback_triggered": False,
                "fallback_reasons": [],
                "qre_json": None,
                "result_json": None,
            }

        dispatch, breakdown, confidence, qre, result = self.quantum.solve(state, problem)

        fallback_reasons: List[str] = []
        fallback_triggered = False
        exec_ms = int(result["backend"]["queue_ms"] + result["backend"]["exec_ms"])
        latency_breach = exec_ms > 1500

        if latency_breach:
            fallback_triggered = True
            fallback_reasons.append("SLA_BREACH")
        if confidence < 0.72:
            fallback_triggered = True
            fallback_reasons.append("LOW_CONFIDENCE")

        if fallback_triggered:
            dispatch, breakdown, confidence = self.classical.solve(state, problem)
            return {
                "route": "FALLBACK_CLASSICAL",
                "dispatch": dispatch,
                "objective_breakdown": breakdown,
                "confidence": confidence,
                "exec_ms": exec_ms,
                "latency_breach": latency_breach,
                "fallback_triggered": True,
                "fallback_reasons": fallback_reasons,
                "qre_json": json.dumps(qre, ensure_ascii=False),
                "result_json": json.dumps(result, ensure_ascii=False),
            }

        return {
            "route": "QUANTUM",
            "dispatch": dispatch,
            "objective_breakdown": breakdown,
            "confidence": confidence,
            "exec_ms": exec_ms,
            "latency_breach": latency_breach,
            "fallback_triggered": False,
            "fallback_reasons": [],
            "qre_json": json.dumps(qre, ensure_ascii=False),
            "result_json": json.dumps(result, ensure_ascii=False),
        }


class EnergyRuntime:
    def __init__(self, scenario: ScenarioName = "economic_dispatch", seed: int = 42):
        self.scenario = scenario
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_id = 0
        self.cumulative_cost_eur = 0.0
        self.records: List[EnergyExecRecord] = []
        self.orchestrator = EnergyHybridOrchestrator(seed=self.seed)
        self.twins: Dict[str, TwinBase] = {}
        self._build_twins()

    def _build_twins(self) -> None:
        ts = utc_now_iso()
        self.twins = {
            "pv_array": PVArrayTwin("pv_array", "pv_array", "PV Array", ts),
            "battery": BatteryTwin("battery", "battery", "Battery", ts),
            "electrolyzer": ElectrolyzerTwin("electrolyzer", "electrolyzer", "Electrolyzer", ts),
            "h2_tank": HydrogenTankTwin("h2_tank", "h2_tank", "H2 Tank", ts),
            "fuel_cell": FuelCellTwin("fuel_cell", "fuel_cell", "Fuel Cell", ts),
            "grid_connection": GridConnectionTwin("grid_connection", "grid_connection", "Grid Connection", ts),
            "critical_load_block": LoadBlockTwin("critical_load_block", "critical_load_block", "Critical Loads", ts, is_critical=True),
            "flex_load_block": LoadBlockTwin("flex_load_block", "flex_load_block", "Flexible Loads", ts, is_critical=False),
            "site_microgrid": SiteMicrogridTwin("site_microgrid", "site_microgrid", "Site Microgrid", ts),
        }

    def _mode_for_scenario(self) -> Mode:
        if self.scenario == "grid_contingency":
            return "resilience"
        if self.scenario == "flex_market":
            return "market"
        return "economic"

    def _generate_base_context(self) -> ScenarioContext:
        mode = self._mode_for_scenario()
        hour = (self.step_id % 288) / 12.0
        solar_wave = max(0.0, np.sin((hour - 6.0) / 12.0 * np.pi))
        irradiance = float(np.clip(0.08 + 0.95 * solar_wave + 0.03 * self.rng.normal(), 0.0, 1.0))
        module_temp = 18.0 + 18.0 * solar_wave + self.rng.normal(scale=1.2)

        crit_load = float(170.0 + 15.0 * np.sin(hour / 24.0 * 2.0 * np.pi + 0.4) + self.rng.normal(scale=4.0))
        flex_load = float(55.0 + 20.0 * np.sin(hour / 24.0 * 2.0 * np.pi - 0.6) + self.rng.normal(scale=5.0))
        crit_load = max(140.0, crit_load)
        flex_load = max(20.0, flex_load)

        import_price = float(0.14 + 0.03 * (hour > 18 or hour < 8) + 0.01 * self.rng.random())
        export_price = float(0.06 + 0.04 * (12 < hour < 17))

        return ScenarioContext(
            scenario=self.scenario,
            mode=mode,
            weather={
                "irradiance_proxy": irradiance,
                "module_temp_proxy": float(module_temp),
            },
            tariff={
                "import_price_eur_kwh": import_price,
                "export_price_eur_kwh": export_price,
            },
            grid_signal={
                "grid_status": "ON",
            },
            loads={
                "critical_load_block": {"requested_kw": crit_load},
                "flex_load_block": {"requested_kw": flex_load},
            },
            active_events=[],
        )

    def _generate_events(self, ctx: ScenarioContext) -> None:
        active: List[ScenarioEvent] = []
        if ctx.weather["irradiance_proxy"] > 0.82:
            active.append(ScenarioEvent("solar_surplus", 0.7, self.step_id, self.step_id, {}))
        if self.step_id % 37 in (0, 1, 2):
            ctx.tariff["import_price_eur_kwh"] += 0.09
            active.append(ScenarioEvent("price_spike", 0.8, self.step_id, self.step_id, {}))
        if self.scenario == "grid_contingency" and self.step_id % 45 in range(10, 18):
            ctx.grid_signal["grid_status"] = "OFF"
            active.append(ScenarioEvent("grid_outage", 1.0, self.step_id, self.step_id, {}))
        if self.scenario == "grid_contingency" and self.step_id % 45 in range(18, 25):
            active.append(ScenarioEvent("brownout", 0.6, self.step_id, self.step_id, {"import_limit_kw": 80.0}))
        if self.scenario == "flex_market" and self.step_id % 28 in range(8, 14):
            active.append(ScenarioEvent("flex_request", 0.7, self.step_id, self.step_id, {"requested_kw": 35.0}))
        ctx.active_events = active

    def get_context(self) -> ScenarioContext:
        ctx = self._generate_base_context()
        self._generate_events(ctx)
        return ctx

    def update_telemetry(self, ctx: ScenarioContext) -> None:
        ts = utc_now_iso()
        for twin in self.twins.values():
            twin.ts = ts

        grid = self.twins["grid_connection"]
        assert isinstance(grid, GridConnectionTwin)
        if any(ev.event_type == "brownout" for ev in ctx.active_events):
            grid.p_import_max_kw = 80.0
        else:
            grid.p_import_max_kw = 250.0

        self.twins["pv_array"].step(5 / 60, ctx.__dict__)
        self.twins["grid_connection"].step(5 / 60, ctx.__dict__)
        self.twins["critical_load_block"].step(5 / 60, ctx.__dict__)
        self.twins["flex_load_block"].step(5 / 60, ctx.__dict__)

    def aggregate_state(self, ctx: ScenarioContext) -> Dict[str, Any]:
        pv = self.twins["pv_array"]
        batt = self.twins["battery"]
        h2 = self.twins["h2_tank"]
        grid = self.twins["grid_connection"]
        crit = self.twins["critical_load_block"]
        flex = self.twins["flex_load_block"]
        assert isinstance(pv, PVArrayTwin)
        assert isinstance(batt, BatteryTwin)
        assert isinstance(h2, HydrogenTankTwin)
        assert isinstance(grid, GridConnectionTwin)
        assert isinstance(crit, LoadBlockTwin)
        assert isinstance(flex, LoadBlockTwin)
        active_event = ctx.active_events[0].event_type if ctx.active_events else None
        return {
            "ts": utc_now_iso(),
            "mode": ctx.mode,
            "scenario": ctx.scenario,
            "active_event": active_event,
            "critical_load_kw": crit.requested_kw,
            "flexible_load_kw": flex.requested_kw,
            "total_load_kw": crit.requested_kw + flex.requested_kw,
            "pv_available_kw": pv.p_available_kw,
            "battery_soc": batt.soc,
            "battery_energy_kwh": batt.energy_kwh,
            "h2_level_kg": h2.level_kg,
            "grid_status": grid.grid_status,
            "import_price_eur_kwh": grid.import_price_eur_kwh,
            "export_price_eur_kwh": grid.export_price_eur_kwh,
        }

    def build_problem(self, state: Dict[str, Any], ctx: ScenarioContext) -> DispatchProblem:
        discrete_vars = 5
        continuous_vars = 4
        event_bonus = 1.2 if ctx.active_events else 0.0
        mode_bonus = 1.5 if ctx.mode == "market" else 0.6 if ctx.mode == "economic" else 0.2
        complexity = discrete_vars * 0.8 + continuous_vars * 0.25 + event_bonus + mode_bonus
        discrete_ratio = discrete_vars / max(discrete_vars + continuous_vars, 1)
        constraints = {
            "power_balance_target_kw": state["total_load_kw"],
            "pv_available_kw": state["pv_available_kw"],
            "battery_soc": state["battery_soc"],
            "battery_energy_kwh": state["battery_energy_kwh"],
            "h2_level_kg": state["h2_level_kg"],
            "grid_status": state["grid_status"],
        }
        objective_terms = {
            "grid_import_cost_weight": state["import_price_eur_kwh"],
            "battery_degradation_weight": 0.015,
            "curtailment_penalty_weight": 0.04,
            "unserved_critical_weight": 5.0,
            "unserved_flex_weight": 0.4,
            "export_revenue_weight": state["export_price_eur_kwh"],
        }
        return DispatchProblem(
            step_id=self.step_id,
            mode=ctx.mode,
            scenario=ctx.scenario,
            objective_name="min_operational_cost_with_resilience",
            constraints=constraints,
            objective_terms=objective_terms,
            complexity_score=complexity,
            discrete_ratio=discrete_ratio,
            horizon_steps=12,
            metadata={"active_event": state["active_event"]},
        )

    def validate_dispatch(self, dispatch: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        dispatch = dict(dispatch)
        if state["grid_status"] != "ON":
            dispatch["grid_import_kw"] = 0.0
            dispatch["grid_export_kw"] = 0.0
        if dispatch.get("battery_charge_kw", 0.0) > 0 and dispatch.get("battery_discharge_kw", 0.0) > 0:
            if dispatch["battery_charge_kw"] >= dispatch["battery_discharge_kw"]:
                dispatch["battery_discharge_kw"] = 0.0
            else:
                dispatch["battery_charge_kw"] = 0.0
        if state["h2_level_kg"] <= 11.0 and dispatch.get("fuel_cell_on", False):
            dispatch["fuel_cell_on"] = False
            dispatch["fuel_cell_power_kw"] = 0.0
        return dispatch

    def apply_dispatch(self, dispatch: Dict[str, Any], dt_h: float) -> None:
        self.twins["pv_array"].apply_dispatch(dispatch, dt_h)
        self.twins["battery"].apply_dispatch(dispatch, dt_h)
        self.twins["electrolyzer"].apply_dispatch(dispatch, dt_h)
        self.twins["fuel_cell"].apply_dispatch(dispatch, dt_h)
        self.twins["grid_connection"].apply_dispatch(dispatch, dt_h)
        self.twins["critical_load_block"].apply_dispatch(dispatch, dt_h)
        self.twins["flex_load_block"].apply_dispatch(dispatch, dt_h)

        self.twins["battery"].step(dt_h, {})
        self.twins["electrolyzer"].step(dt_h, {})
        self.twins["fuel_cell"].step(dt_h, {})

        elz = self.twins["electrolyzer"]
        fc = self.twins["fuel_cell"]
        assert isinstance(elz, ElectrolyzerTwin)
        assert isinstance(fc, FuelCellTwin)
        self.twins["h2_tank"].step(
            dt_h,
            {
                "h2_produced_kg": elz.h2_production_kgph * dt_h,
                "h2_consumed_kg": fc.h2_consumption_kgph * dt_h,
            },
        )

    def compute_record(self, ctx: ScenarioContext, state: Dict[str, Any], decision: Dict[str, Any]) -> EnergyExecRecord:
        pv = self.twins["pv_array"]
        batt = self.twins["battery"]
        elz = self.twins["electrolyzer"]
        h2 = self.twins["h2_tank"]
        fc = self.twins["fuel_cell"]
        grid = self.twins["grid_connection"]
        crit = self.twins["critical_load_block"]
        flex = self.twins["flex_load_block"]
        assert isinstance(pv, PVArrayTwin)
        assert isinstance(batt, BatteryTwin)
        assert isinstance(elz, ElectrolyzerTwin)
        assert isinstance(h2, HydrogenTankTwin)
        assert isinstance(fc, FuelCellTwin)
        assert isinstance(grid, GridConnectionTwin)
        assert isinstance(crit, LoadBlockTwin)
        assert isinstance(flex, LoadBlockTwin)

        total_load = crit.requested_kw + flex.requested_kw
        renewable_share = min(1.0, pv.p_dispatched_kw / total_load) if total_load > 1e-6 else 0.0
        h2_equiv_kwh = max(0.0, h2.level_kg - h2.min_reserve_kg) * fc.eta_electric * fc.h2_lhv_kwhkg
        resilience_margin_h = (max(0.0, batt.energy_kwh - batt.reserve_locked_kwh) + h2_equiv_kwh) / max(crit.requested_kw, 1e-6)

        obj = decision["objective_breakdown"]
        operating_cost = (
            obj["grid_import_cost"]
            + obj["battery_degradation_cost"]
            + obj["electrolyzer_startup_cost"]
            + obj["fuel_cell_startup_cost"]
            + obj["curtailment_penalty"]
            + obj["unserved_critical_penalty"]
            + obj["unserved_flex_penalty"]
            + obj["export_revenue"]
        )
        self.cumulative_cost_eur += operating_cost

        return EnergyExecRecord(
            step_id=self.step_id,
            ts=utc_now_iso(),
            mode=ctx.mode,
            scenario=ctx.scenario,
            active_event=state["active_event"],
            total_load_kw=total_load,
            critical_load_kw=crit.requested_kw,
            flexible_load_kw=flex.requested_kw,
            pv_available_kw=pv.p_available_kw,
            pv_used_kw=pv.p_dispatched_kw,
            pv_curtailed_kw=pv.curtailment_kw,
            battery_soc=batt.soc,
            battery_charge_kw=batt.p_charge_kw,
            battery_discharge_kw=batt.p_discharge_kw,
            battery_energy_kwh=batt.energy_kwh,
            electrolyzer_on=elz.status_on,
            electrolyzer_power_kw=elz.p_consumption_kw,
            h2_production_kgph=elz.h2_production_kgph,
            h2_level_kg=h2.level_kg,
            h2_fill_ratio=h2.level_kg / max(h2.capacity_kg, 1e-6),
            fuel_cell_on=fc.status_on,
            fuel_cell_power_kw=fc.p_output_kw,
            fuel_cell_h2_use_kgph=fc.h2_consumption_kgph,
            grid_import_kw=grid.p_import_kw,
            grid_export_kw=grid.p_export_kw,
            import_price_eur_kwh=grid.import_price_eur_kwh,
            export_price_eur_kwh=grid.export_price_eur_kwh,
            renewable_share=renewable_share,
            resilience_margin_h=resilience_margin_h,
            operating_cost_eur_step=operating_cost,
            cumulative_cost_eur=self.cumulative_cost_eur,
            curtailment_kw=pv.curtailment_kw,
            unserved_critical_kw=crit.unserved_kw,
            unserved_flex_kw=flex.unserved_kw,
            decision_route=decision["route"],
            decision_confidence=decision["confidence"],
            exec_ms=decision["exec_ms"],
            latency_breach=decision["latency_breach"],
            fallback_triggered=decision["fallback_triggered"],
            fallback_reasons=decision["fallback_reasons"],
            complexity_score=float(decision["problem"].complexity_score),
            discrete_ratio=float(decision["problem"].discrete_ratio),
            qre_json=decision["qre_json"],
            result_json=decision["result_json"],
            dispatch_json=json.dumps(decision["dispatch"], ensure_ascii=False),
            objective_breakdown_json=json.dumps(decision["objective_breakdown"], ensure_ascii=False),
        )

    def step(self, dt_h: float = 5.0 / 60.0) -> EnergyExecRecord:
        self.step_id += 1
        ctx = self.get_context()
        self.update_telemetry(ctx)
        state = self.aggregate_state(ctx)
        problem = self.build_problem(state, ctx)
        decision = self.orchestrator.solve(state, problem)
        decision["problem"] = problem
        dispatch = self.validate_dispatch(decision["dispatch"], state)
        decision["dispatch"] = dispatch
        self.apply_dispatch(dispatch, dt_h)
        record = self.compute_record(ctx, state, decision)
        self.records.append(record)
        return record

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])

    def latest_state(self) -> Dict[str, Any]:
        return asdict(self.records[-1]) if self.records else {}

    def twin_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.snapshot() for k, v in self.twins.items()}

    def reset(self) -> None:
        scenario = self.scenario
        seed = self.seed
        self.__init__(scenario=scenario, seed=seed)


def run_demo(steps: int = 48, scenario: ScenarioName = "economic_dispatch", seed: int = 42) -> pd.DataFrame:
    rt = EnergyRuntime(scenario=scenario, seed=seed)
    for _ in range(steps):
        rt.step()
    return rt.dataframe()


if __name__ == "__main__":
    df = run_demo(steps=24, scenario="economic_dispatch", seed=42)
    print(df.tail(5).to_string(index=False))
