# PIRL

Recurrent PPO local navigation with **physics-informed (HJB) critic regularization**, trained in Isaac Lab and exported to ONNX.

> Thesis: *Physics-Informed Regularization for PPO-RNN in Autonomous Obstacle Avoidance under Partial Observability* — Ulan Sharipov, Nazarbayev University, 2026.

Architecture, CLI options, ONNX export, configuration, validation, and troubleshooting: [AGENTS.md](AGENTS.md).

---

## Prerequisites

PIRL pins [Isaac Lab v2.3.2](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2) as a Git submodule. That release uses Isaac Sim 5.1.0. Docker Engine with Compose v2 and an NVIDIA GPU runtime are required.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/ulanpy/pirl.git
cd pirl
```

For an existing clone:

```bash
git submodule update --init --recursive
```

### 2. Start the Container (host)

From the `pirl/` directory, start the pinned Isaac Lab Compose stack directly. This is deliberately headless: PIRL uses WebRTC/livestream when remote visualization is needed, so it does not require X11 or `$DISPLAY`.

```bash
PIRL_PROJECT_DIR="$(pwd)" \
docker compose \
  --file third_party/IsaacLab/docker/docker-compose.yaml \
  --file docker-compose.overlay.yaml \
  --profile base \
  --env-file third_party/IsaacLab/docker/.env.base \
  up --detach --remove-orphans
```

The first run pulls the pinned official image; later starts are fast. Container name: `isaac-lab-base`. Isaac Lab and PIRL are mounted separately at `/workspace/isaaclab` and `/workspace/pirl`.

### 3. Enter the Container (host)

```bash
docker exec -it isaac-lab-base bash
cd /workspace/pirl
```

### 4. Install the PIRL Package (container)

Once per environment (Isaac Lab image already includes skrl 2.x; editable install also declares `skrl>=2.1.0`):

```bash
python -m pip install -e source/pirl
python scripts/list_envs.py   # expect burger
```

### 5. Run a Trained Agent (Playback)

```bash
python scripts/skrl/play.py \
  --task=burger \
  --agent=skrl_ppo_aux_cfg_entry_point \
  --checkpoint=logs/skrl/burger_direct/2026-06-04_14-09-41_ppo_aux_torch/ \
  --livestream 2
```

### 6. Train a New Policy

```bash
python scripts/skrl/train.py --task=burger
```

Training takes ~1-2 hours on a RTX 4090. Logs go to `logs/skrl/burger_direct/`.

### 7. Monitor Training (TensorBoard, container)

```bash
tensorboard --logdir logs/skrl/burger_direct --bind_all --port 6006
```

Open `http://localhost:6006` from the host if port 6006 is exposed in Docker. `--bind_all` listens on all interfaces (for access outside the container); do not combine it with `--host`.

---

## Documentation

| Topic | Doc |
| --- | --- |
| Architecture, deployment, CLI, troubleshooting | [AGENTS.md](AGENTS.md) |
| Network diagrams | [docs/ppo_aux_architecture_graph.md](docs/ppo_aux_architecture_graph.md) |
| Task definitions & rewards | [docs/environment.md](docs/environment.md) |
| ONNX / observation schema (V2.1) | [docs/DEPLOYMENT_OBSERVATION_SPACE.md](docs/DEPLOYMENT_OBSERVATION_SPACE.md) |
| ROS2 path manager contract | [docs/pirl_path_contract_ros_like.md](docs/pirl_path_contract_ros_like.md) |
| HJB regularizer theory | [docs/HJB_THEORY_TIME_DISTANCE.md](docs/HJB_THEORY_TIME_DISTANCE.md) |

---

## License

[MIT License](LICENSE)
