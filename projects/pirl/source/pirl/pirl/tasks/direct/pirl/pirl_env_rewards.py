import torch


def compute_reward(
    forward_speed: torch.Tensor,
    alignment: torch.Tensor,
    has_collision: torch.Tensor,
    commands: torch.Tensor,
    cfg,
) -> torch.Tensor:
    # Reward = Speed * exp(Alignment) - Penalty
    reward = forward_speed * torch.exp(alignment)
    reward += has_collision.float() * cfg.rew_scale_collision
    # Penalize reverse motion (no rear lidar coverage)
    reverse_speed = torch.clamp(-forward_speed, min=0.0)
    reward += reverse_speed * cfg.rew_scale_reverse
    # Penalize standing still when a command exists
    cmd_norm = torch.linalg.norm(commands, dim=-1, keepdim=True)
    standstill = (forward_speed.abs() < cfg.standstill_speed_threshold) & (cmd_norm > 0.1)
    reward += standstill.float() * cfg.rew_scale_standstill
    return reward
