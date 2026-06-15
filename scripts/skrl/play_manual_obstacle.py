#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint and manually move one dynamic obstacle slot.

Keyboard controls (local env frame):
  I / K : move +X / -X
  L / J : move +Y / -Y
  U / O : rotate +yaw / -yaw
  P     : reset obstacle pose to initial
"""

import argparse
import math
import os
import sys
import random
import time
import importlib

import carb
from isaaclab.app import AppLauncher

# Default public IP advertised to WebRTC clients when --livestream=1 is used.
DEFAULT_PUBLIC_IP = "100.118.210.31"

parser = argparse.ArgumentParser(description="Play skrl checkpoint with manual obstacle control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length (steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="skrl_ppo_aux_cfg_entry_point",
    help=(
        "Agent config entry point. Defaults to skrl_ppo_aux_cfg_entry_point so "
        "ppo_aux checkpoints load with matching model definitions."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use pre-trained checkpoint from Nucleus.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time when possible.")
parser.add_argument(
    "--public_ip",
    type=str,
    default=None,
    help=(
        "Public IP advertised to WebRTC client when using --livestream=1. "
        f"If unset: PUBLIC_IP env var, then {DEFAULT_PUBLIC_IP}."
    ),
)
parser.add_argument("--manual_env_id", type=int, default=0, help="Environment id of manually controlled obstacle.")
parser.add_argument("--manual_slot_id", type=int, default=0, help="Obstacle slot id to control.")
parser.add_argument("--manual_speed", type=float, default=0.75, help="Manual XY speed in m/s.")
parser.add_argument("--manual_yaw_speed", type=float, default=1.5, help="Manual yaw speed in rad/s.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

if getattr(args_cli, "livestream", -1) == 1:
    resolved_public_ip = args_cli.public_ip or os.environ.get("PUBLIC_IP") or DEFAULT_PUBLIC_IP
    os.environ["PUBLIC_IP"] = resolved_public_ip
    print(f"[INFO][play_manual_obstacle.py]: livestream=1, PUBLIC_IP={resolved_public_ip}")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import skrl
import torch
from packaging import version

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import pirl.tasks  # noqa: F401
from pirl.tasks.direct.pirl.agents.runner_utils import get_runner

SKRL_VERSION = "2.1.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. Install skrl>={SKRL_VERSION}."
    )
    raise SystemExit(1)


class ManualObstacleController:
    """Keyboard-driven obstacle local pose integrator."""

    def __init__(
        self,
        env_cfg,
        local_xy: torch.Tensor,
        yaw: float,
        speed: float,
        yaw_speed: float,
    ) -> None:
        self.xy = local_xy.clone().float()
        self.initial_xy = local_xy.clone().float()
        self.yaw = float(yaw)
        self.initial_yaw = float(yaw)
        self.speed = float(speed)
        self.yaw_speed = float(yaw_speed)
        self.x_min = float(env_cfg.dyn_obstacle_xy_range[0][0])
        self.x_max = float(env_cfg.dyn_obstacle_xy_range[0][1])
        self.y_min = float(env_cfg.dyn_obstacle_xy_range[1][0])
        self.y_max = float(env_cfg.dyn_obstacle_xy_range[1][1])
        self._pressed: set[carb.input.KeyboardInput] = set()
        self._sub = None

        appwindow_module = None
        for mod_name in ("omni.appwindow", "omni.kit.appwindow"):
            try:
                appwindow_module = importlib.import_module(mod_name)
                break
            except ModuleNotFoundError:
                continue
        if appwindow_module is None:
            raise RuntimeError(
                "Keyboard control requires omni appwindow module "
                "(tried omni.appwindow and omni.kit.appwindow)."
            )
        appwindow = appwindow_module.get_default_app_window()
        keyboard = appwindow.get_keyboard()
        input_iface = carb.input.acquire_input_interface()
        self._sub = input_iface.subscribe_to_keyboard_events(keyboard, self._on_key_event)

    def _on_key_event(self, event: carb.input.KeyboardEvent) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(event.input)
            if event.input == carb.input.KeyboardInput.P:
                self.xy = self.initial_xy.clone()
                self.yaw = self.initial_yaw
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(event.input)
        return True

    def close(self) -> None:
        if self._sub is not None:
            carb.input.acquire_input_interface().unsubscribe_to_keyboard_events(self._sub)
            self._sub = None

    def update(self, dt: float) -> tuple[torch.Tensor, float]:
        # Local frame commands
        move_x = 0.0
        move_y = 0.0
        turn = 0.0
        if carb.input.KeyboardInput.I in self._pressed:
            move_x += 1.0
        if carb.input.KeyboardInput.K in self._pressed:
            move_x -= 1.0
        if carb.input.KeyboardInput.L in self._pressed:
            move_y += 1.0
        if carb.input.KeyboardInput.J in self._pressed:
            move_y -= 1.0
        if carb.input.KeyboardInput.U in self._pressed:
            turn += 1.0
        if carb.input.KeyboardInput.O in self._pressed:
            turn -= 1.0

        self.xy[0] += float(dt) * self.speed * move_x
        self.xy[1] += float(dt) * self.speed * move_y
        self.xy[0] = torch.clamp(self.xy[0], min=self.x_min, max=self.x_max)
        self.xy[1] = torch.clamp(self.xy[1], min=self.y_min, max=self.y_max)
        self.yaw += float(dt) * self.yaw_speed * turn
        return self.xy, self.yaw


if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_manual_obstacle"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Access raw env before skrl wrapper.
    raw_env = env.unwrapped
    if not hasattr(raw_env, "dyn_obstacles") or raw_env.dyn_obstacles is None:
        raise RuntimeError(
            "Task has no dynamic obstacles enabled. Set dyn_obstacle_enabled=True in env cfg."
        )
    dyn = raw_env.dyn_obstacles
    env_id = int(args_cli.manual_env_id)
    slot_id = int(args_cli.manual_slot_id)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = get_runner(env, experiment_cfg, args_cli.ml_framework)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    obs, _ = env.reset()
    states = env.state()

    # Initialize manual obstacle pose from current slot state.
    with torch.no_grad():
        phase = dyn._phase[env_id, slot_id]
        local_xy = dyn._anchor_xy[env_id, slot_id] + torch.tensor(
            [
                dyn._radius[env_id, slot_id] * torch.cos(phase),
                dyn._radius[env_id, slot_id] * torch.sin(phase),
            ],
            device=phase.device,
            dtype=torch.float32,
        )
        yaw = float(phase.item() + 0.5 * math.pi)

    controller = ManualObstacleController(
        env_cfg=raw_env.cfg,
        local_xy=local_xy.detach().cpu(),
        yaw=yaw,
        speed=args_cli.manual_speed,
        yaw_speed=args_cli.manual_yaw_speed,
    )
    print(
        "[INFO] Manual obstacle controls: I/K (+/-X), L/J (+/-Y), U/O (+/-yaw), P reset"
    )
    print(f"[INFO] Controlling obstacle slot {slot_id} in env {env_id}")

    timestep = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()

            manual_xy_cpu, manual_yaw = controller.update(dt)
            manual_xy = manual_xy_cpu.to(device=raw_env.device, dtype=torch.float32)
            dyn.set_manual_obstacle_pose(env_id=env_id, slot_id=slot_id, local_xy=manual_xy, yaw=manual_yaw)

            with torch.inference_mode():
                outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, _ = env.step(actions)
                states = env.state()

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        controller.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

