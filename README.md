# Hybrid Quantum-Classical Control Room — Energy Vertical

This repository contains a Streamlit prototype verticalized for energy-system operation.

## What it simulates

A hydrogen-enabled microgrid with:
- PV array
- Battery energy storage
- Electrolyzer
- H2 tank
- Fuel cell
- Grid connection
- Critical and flexible loads

The application uses a hybrid routing logic:
- classical route for safe, low-latency or low-complexity decisions
- mock quantum route for higher-discreteness dispatch problems
- fallback-to-classical when latency or confidence become unacceptable

## Files

- `app.py`: Streamlit dashboard
- `energy_runtime.py`: energy simulation engine and hybrid orchestration logic
- `requirements.txt`: Python dependencies
- `Dockerfile`: container image definition
- `docker-compose.yml`: local container execution

## Local run

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Docker run

```bash
docker compose up --build
```

Then open:

`http://localhost:8501`

## Notes

This prototype is a verticalized demo, not a certified operational EMS.
The quantum path is simulated to demonstrate orchestration, auditability and UI behavior.
