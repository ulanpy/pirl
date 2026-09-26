from __future__ import annotations

import torch


def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length - 1


def collision(env) -> torch.Tensor:
    env.refresh_navigation_state()
    return env._latest_lidar_ranges_m.min(dim=1).values < float(env.task_cfg.collision_robot_radius)


def success(env) -> torch.Tensor:
    env.refresh_navigation_state()
    return env.final_goal_dist.squeeze(-1) < float(env.task_cfg.path_goal_threshold)
