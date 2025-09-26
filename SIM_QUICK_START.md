# Aloha Sim Quick Start

## Setup Environment
```bash
source examples/aloha_sim/.venv/bin/activate
```

## Start Simulation (Terminal 1)
```bash
MUJOCO_GL=egl python3 examples/aloha_sim/main.py
```

## Start Policy Server (Terminal 2)
```bash
uv run scripts/serve_policy.py --env ALOHA_SIM
```
