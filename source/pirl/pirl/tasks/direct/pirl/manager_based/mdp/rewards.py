from __future__ import annotations

import torch


def _state(env) -> None:
    env.refresh_navigation_state()


def progress(env) -> torch.Tensor:
    _state(env)
    value = (env.curr_path_s - env.prev_path_s).squeeze(-1)
    env.reward_components[:, 0] = value * float(env.task_cfg.rew_scale_progress)
    env.prev_path_s.copy_(env.curr_path_s)
    return value


def path_error(env) -> torch.Tensor:
    _state(env)
    value = -(env.curr_path_error.square()).squeeze(-1)
    env.reward_components[:, 1] = value * float(env.task_cfg.rew_scale_path_error)
    return value


def heading(env) -> torch.Tensor:
    _state(env)
    speed = env.robot.data.root_com_lin_vel_b[:, 0:1]
    gate = torch.clamp(speed / float(env.task_cfg.max_lin_vel), 0.0, 1.0)
    value = (env.path_heading_cos * gate).squeeze(-1)
    env.reward_components[:, 2] = value * float(env.task_cfg.rew_scale_heading)
    return value


def collision(env) -> torch.Tensor:
    _state(env)
    value = (env._latest_lidar_ranges_m.min(dim=1).values < float(env.task_cfg.collision_robot_radius)).float()
    env.reward_components[:, 3] = value * float(env.task_cfg.rew_scale_collision)
    return value


def time(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def success(env) -> torch.Tensor:
    _state(env)
    return (env.final_goal_dist.squeeze(-1) < float(env.task_cfg.path_goal_threshold)).float()
