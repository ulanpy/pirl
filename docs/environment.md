# Environment Setup for PIRL Project

This document describes the environment configuration for the Physics Informed Deep RL (PIRL) project focused on dynamic obstacle avoidance for a caterpillar AGV.

## Stack Overview
- **Simulator:** NVIDIA Isaac Sim 5.1.0
- **Framework:** Isaac Lab (latest main branch)
- **RL Library:** [skrl](https://skrl.readthedocs.io/) (optimized for Isaac Sim/Lab)
- **Algorithm:** SAC (Soft Actor-Critic)
- **Containerization:** Docker with NVIDIA Container Toolkit

## Docker Configuration

### Base Image
We use `nvcr.io/nvidia/isaac-sim:5.1.0` as the base image. Isaac Lab is installed during the build process to ensure compatibility.

### Key Dependencies
- `ncurses-term`: Required for proper terminal handling during Isaac Lab installation.
- `skrl`, `wandb`, `onnx`: Installed via `pip` within the Isaac Lab environment.

## Usage Instructions

### Building the Environment
```bash
docker-compose build
```

### Running the Container
```bash
docker-compose up -d
docker exec -it isaac_lab_pinn bash
```

### Training (Headless)
To run training in headless mode (recommended for maximum performance):
```bash
./isaaclab.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Velocity-Caterpillar-v0 --headless
```

### Debugging with Livestream (WebRTC)
This is the **primary method** for remote or headless debugging:
1. Run the script **without** the `--headless` flag inside the container:
   ```bash
   isaaclab -p src/train.py --task Isaac-Velocity-Caterpillar-v0
   ```
2. Open a browser on the host machine at `http://localhost:8211`.
3. Wait for the message `Isaac Sim Full Streaming App is loaded`.

### Debugging with Native GUI (X11)
Use this for **local debugging** with zero latency (requires a local Linux machine):
1. On your host machine, allow X11 connections:
   ```bash
   xhost +local:
   ```
2. Run Isaac Sim with GUI from the container:
   ```bash
   docker exec -it isaac_lab_pinn /isaac-sim/runapp.sh
   ```

## Useful Documentation Links
- [Isaac Lab Container Deployment Guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html)
- [Isaac Sim 5.1.0 Container Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)
- [Isaac Lab Documentation Home](https://isaac-sim.github.io/IsaacLab/)
- [skrl (Reinforcement Learning Library)](https://skrl.readthedocs.io/)

## Directory Structure
- `/workspace/IsaacLab`: Root directory of Isaac Lab.
- `/workspace/IsaacLab/source/extensions/pinn_nav`: Mounted local `pirl` directory for custom development.
- `./docker_data`: Local directory for persistent storage (cache, logs, data).

## 
xhost +local: && docker exec -it isaac_lab_pinn _isaac_sim/runapp.sh