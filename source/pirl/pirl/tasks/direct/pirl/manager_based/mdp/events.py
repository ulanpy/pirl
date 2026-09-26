from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def reset_navigation(env, env_ids: Sequence[int]) -> None:
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return
    cfg = env.task_cfg
    root = env.robot.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    radius = float(cfg.robot_spawn_radius) * torch.sqrt(torch.rand(len(ids), device=env.device))
    a0, a1 = cfg.spawn_angle_range
    angle = torch.rand(len(ids), device=env.device) * (a1 - a0) + a0
    root[:, 0] += radius * torch.cos(angle)
    root[:, 1] += radius * torch.sin(angle)
    half = 0.5 * (angle + math.pi)
    root[:, 3], root[:, 4], root[:, 5], root[:, 6] = torch.cos(half), 0.0, 0.0, torch.sin(half)
    env.robot.write_root_state_to_sim(root, ids)
    env.dyn_obstacles.reset(ids, env.scene.env_origins)
    env.lidar.reset(ids.tolist())
    env.costmap.reset(ids.tolist())
    env.path_manager.reset(ids.tolist(), env.scene.env_origins[ids, :2])
    points = env.path_manager.path_points_w[ids]
    pos = root[:, :2]
    d2 = torch.sum((points - pos.unsqueeze(1)).square(), dim=-1)
    nearest = torch.argmin(d2, dim=1)
    s = env.path_manager.path_s[ids, nearest].unsqueeze(-1)
    env.path_manager.path_idx[ids] = nearest
    env.prev_path_s[ids] = s
    env.curr_path_s[ids] = s
    env.prev_reward_components[ids] = 0.0
    env.reward_components[ids] = 0.0
    env.action_manager.get_term("drive").prev_actions[ids] = 0.0
    env.invalidate_navigation_state()
