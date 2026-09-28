"""Viewport helpers for interactive PIRL teleoperation."""

import torch
import isaaclab.utils.math as math_utils


def update_robot_ground_view(env) -> None:
    """Follow the Burger from above and behind while looking at nearby ground."""
    robot_pos = env.robot.data.root_pos_w[0]
    robot_quat = env.robot.data.root_quat_w[0:1]
    forward = math_utils.quat_apply(
        robot_quat, torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
    )[0]
    eye = robot_pos - 0.35 * forward + torch.tensor([0.0, 0.0, 0.80], device=env.device)
    target = robot_pos + 0.80 * forward + torch.tensor([0.0, 0.0, 0.03], device=env.device)
    env.sim.set_camera_view(eye=eye.detach().cpu().numpy(), target=target.detach().cpu().numpy())
