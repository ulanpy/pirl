from __future__ import annotations

import torch


def vector(env) -> torch.Tensor:
    env.refresh_navigation_state()
    action = env.action_manager.get_term("drive")
    robot_pos = env.robot.data.root_pos_w[:, :2]
    path = env.path_manager.get_resampled_segment(robot_pos, env.robot.data.root_quat_w, env.curr_path_s)
    ego = torch.stack((env.robot.data.root_com_lin_vel_b[:, 0], env.robot.data.root_com_ang_vel_b[:, 2]), dim=-1)
    tracking = torch.cat((env.curr_path_error_signed, env.heading_error), dim=-1)
    env.prev_reward_components.copy_(torch.clamp(env.reward_components, -1.0, 1.0))
    return torch.cat((ego, tracking, path, action.prev_actions, env.prev_reward_components), dim=-1)


def costmap(env) -> torch.Tensor:
    env.refresh_navigation_state()
    return env.current_costmap
