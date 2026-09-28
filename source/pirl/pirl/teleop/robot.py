"""Human gamepad input mapped to the PIRL robot action contract."""

import torch

from isaaclab.devices import Se2Gamepad, Se2GamepadCfg


class GamepadRobotController:
    """Map gamepad forward/yaw commands to normalized Burger actions."""

    def __init__(self) -> None:
        self.gamepad = Se2Gamepad(Se2GamepadCfg(sim_device="cpu"))

    def close(self) -> None:
        del self.gamepad

    def update(self) -> torch.Tensor:
        command = self.gamepad.advance().cpu()
        # Burger is differential drive; lateral SE(2) input is intentionally ignored.
        # Carb's right-stick sign is opposite to the Burger yaw-action convention.
        return torch.stack((command[0], -command[2])).clamp(-1.0, 1.0)
