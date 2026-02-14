# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate one or more skrl checkpoints: run N episodes and report mean ± std return.
Use for quantitative comparison between runs (e.g. grid 1.6m vs 3m, or different seeds).

Example:
  python scripts/skrl/eval.py --task Template-Pirl-Direct-v0 --checkpoint logs/skrl/jettank_direct/RUN_DIR/checkpoints/best_agent.pt --num_episodes 50 --num_envs 4
  # Compare two runs:
  python scripts/skrl/eval.py --task Template-Pirl-Direct-v0 --checkpoint path/to/run1/best_agent.pt --checkpoint path/to/run2/best_agent.pt --num_episodes 30
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate skrl checkpoint(s): report mean ± std episode return.")
parser.add_argument("--task", type=str, default="Template-Pirl-Direct-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, action="append", required=True, help="Path to checkpoint .pt (can pass multiple).")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes per checkpoint.")
parser.add_argument("--num_envs", type=int, default=4, help="Envs to run in parallel (faster).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import re
import copy
import torch
import gymnasium as gym
import numpy as np
from packaging import version
import skrl

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. pip install skrl>={SKRL_VERSION}")
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.assets import retrieve_file_path

import pirl.tasks  # noqa: F401

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


def _parse_run_env_yaml(env_yaml_path: str) -> dict | None:
    """Read only scalar key: value lines from env.yaml (avoids !!python/tuple etc.)."""
    out = {}
    # Match "key: number" or "key: true/false" at start of line (with optional indent)
    pattern = re.compile(r"^\s*(\w+):\s*([-+]?\d*\.?\d+|\d+)\s*$")
    with open(env_yaml_path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                key, val = m.group(1), m.group(2)
                if key in ("grid_size_m", "grid_resolution", "grid_history_len", "grid_history_interval_steps", "path_segment_len", "grid_width_cells"):
                    try:
                        out[key] = int(val) if "." not in val else float(val)
                    except ValueError:
                        out[key] = float(val)
    return out if out else None


def apply_run_env_config(env_cfg, run_dir: str) -> bool:
    """If run_dir/params/env.yaml exists, apply grid_size_m, grid_resolution, grid_history_len,
    path_segment_len to env_cfg and recompute grid_width_cells and observation_space.
    Returns True if applied (so observation space matches the run that produced the checkpoint).
    """
    env_yaml_path = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(env_yaml_path):
        return False
    run_env = _parse_run_env_yaml(env_yaml_path)
    if not run_env:
        return False
    # Apply costmap/path params that change observation shape
    if "grid_size_m" in run_env:
        env_cfg.grid_size_m = run_env["grid_size_m"]
    if "grid_resolution" in run_env:
        env_cfg.grid_resolution = run_env["grid_resolution"]
    if "grid_history_len" in run_env:
        env_cfg.grid_history_len = run_env["grid_history_len"]
    if "grid_history_interval_steps" in run_env:
        env_cfg.grid_history_interval_steps = run_env["grid_history_interval_steps"]
    if "path_segment_len" in run_env:
        env_cfg.path_segment_len = run_env["path_segment_len"]
    env_cfg.grid_width_cells = int(round(env_cfg.grid_size_m / env_cfg.grid_resolution))
    vec_dim = 3 + (env_cfg.path_segment_len * 2)
    env_cfg.observation_space = gym.spaces.Dict(
        {
            "vec": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(vec_dim,),
                dtype=np.float32,
            ),
            "costmap": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(env_cfg.grid_history_len, env_cfg.grid_width_cells, env_cfg.grid_width_cells),
                dtype=np.float32,
            ),
        }
    )
    return True


def _to_np(x):
    """Convert tensor (possibly CUDA) or array to numpy."""
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def run_episodes(env, runner, num_episodes: int):
    """Run num_episodes (across num_envs), return list of episode returns."""
    n_envs = env.unwrapped.num_envs
    episode_returns = []
    episode_rewards = np.zeros(n_envs)
    obs, _ = env.reset()
    finished = 0
    while finished < num_episodes:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
        obs, rewards, terminated, truncated, _ = env.step(actions)
        r = _to_np(rewards).flatten()
        episode_rewards += r
        term = _to_np(terminated).flatten()
        trunc = _to_np(truncated).flatten()
        dones = np.logical_or(term, trunc)
        for i in range(n_envs):
            if dones[i]:
                episode_returns.append(float(episode_rewards[i]))
                episode_rewards[i] = 0.0
                finished += 1
                if finished >= num_episodes:
                    break
    return episode_returns


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectRLEnvCfg | ManagerBasedRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    agent_cfg["seed"] = args_cli.seed
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    results = []
    for ckpt_path in args_cli.checkpoint:
        path = retrieve_file_path(ckpt_path) if os.path.isfile(ckpt_path) else ckpt_path
        if not os.path.isfile(path):
            print(f"[WARN] Checkpoint not found: {path}")
            results.append((path, None))
            continue
        # Use env config from the run that produced this checkpoint (so observation space matches)
        run_dir = os.path.dirname(os.path.dirname(path))
        cfg = copy.deepcopy(env_cfg)
        cfg.scene.num_envs = args_cli.num_envs
        cfg.seed = args_cli.seed
        cfg.log_dir = os.path.abspath("logs/skrl/eval")
        if apply_run_env_config(cfg, run_dir):
            w = getattr(cfg, "grid_width_cells", None)
            print(f"[INFO] Using run config from {run_dir} (grid {w}x{w})")
        env = gym.make(args_cli.task, cfg=cfg)
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
        runner = Runner(env, agent_cfg)
        runner.agent.load(path)
        runner.agent.set_running_mode("eval")
        returns = run_episodes(env, runner, args_cli.num_episodes)
        env.close()
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        results.append((path, (mean_r, std_r, returns)))
        print(f"  {path}\n    return: {mean_r:.2f} ± {std_r:.2f}  (n={len(returns)})")

    print("\n--- Summary ---")
    for path, res in results:
        if res is None:
            print(f"  {path}: (failed to load)")
        else:
            mean_r, std_r, _ = res
            print(f"  {path}\n    mean_return: {mean_r:.2f} ± {std_r:.2f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
