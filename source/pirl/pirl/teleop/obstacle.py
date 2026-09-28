"""Interactive controllers for a kinematic obstacle in local environment coordinates."""

import importlib

import carb
import torch
from isaaclab.devices import Se2Gamepad, Se2GamepadCfg


class KeyboardObstacleController:
    """Keyboard-driven obstacle local-pose integrator."""

    def __init__(self, env_cfg, local_xy: torch.Tensor, yaw: float, speed: float, yaw_speed: float) -> None:
        self.xy = local_xy.clone().float()
        self.initial_xy = local_xy.clone().float()
        self.yaw = float(yaw)
        self.initial_yaw = float(yaw)
        self.speed = float(speed)
        self.yaw_speed = float(yaw_speed)
        self.x_min = float(env_cfg.obstacles.xy_range[0][0])
        self.x_max = float(env_cfg.obstacles.xy_range[0][1])
        self.y_min = float(env_cfg.obstacles.xy_range[1][0])
        self.y_max = float(env_cfg.obstacles.xy_range[1][1])
        self._pressed: set[carb.input.KeyboardInput] = set()

        appwindow_module = next(
            (module for name in ("omni.appwindow", "omni.kit.appwindow") if (module := self._import_optional(name))),
            None,
        )
        if appwindow_module is None:
            raise RuntimeError("Keyboard control requires an Omni appwindow module.")
        keyboard = appwindow_module.get_default_app_window().get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._subscription = self._input.subscribe_to_keyboard_events(keyboard, self._on_key_event)

    @staticmethod
    def _import_optional(name: str):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            return None

    def _on_key_event(self, event: carb.input.KeyboardEvent) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(event.input)
            if event.input == carb.input.KeyboardInput.P:
                self.reset()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(event.input)
        return True

    def reset(self) -> None:
        self.xy = self.initial_xy.clone()
        self.yaw = self.initial_yaw

    def close(self) -> None:
        self._input.unsubscribe_to_keyboard_events(self._subscription)

    def update(self, dt: float) -> tuple[torch.Tensor, float]:
        move_x = float(carb.input.KeyboardInput.I in self._pressed) - float(carb.input.KeyboardInput.K in self._pressed)
        move_y = float(carb.input.KeyboardInput.L in self._pressed) - float(carb.input.KeyboardInput.J in self._pressed)
        turn = float(carb.input.KeyboardInput.U in self._pressed) - float(carb.input.KeyboardInput.O in self._pressed)
        self.xy[0] += float(dt) * self.speed * move_x
        self.xy[1] += float(dt) * self.speed * move_y
        self.xy[0] = torch.clamp(self.xy[0], min=self.x_min, max=self.x_max)
        self.xy[1] = torch.clamp(self.xy[1], min=self.y_min, max=self.y_max)
        self.yaw += float(dt) * self.yaw_speed * turn
        return self.xy, self.yaw


class GamepadObstacleController:
    """Integrate Isaac Lab SE(2) gamepad commands into an obstacle local pose."""

    def __init__(self, env_cfg, local_xy: torch.Tensor, yaw: float, speed: float, yaw_speed: float) -> None:
        self.xy = local_xy.clone().float()
        self.initial_xy = local_xy.clone().float()
        self.yaw = float(yaw)
        self.initial_yaw = float(yaw)
        self.x_min = float(env_cfg.obstacles.xy_range[0][0])
        self.x_max = float(env_cfg.obstacles.xy_range[0][1])
        self.y_min = float(env_cfg.obstacles.xy_range[1][0])
        self.y_max = float(env_cfg.obstacles.xy_range[1][1])
        self.gamepad = Se2Gamepad(
            Se2GamepadCfg(
                v_x_sensitivity=float(speed),
                v_y_sensitivity=float(speed),
                omega_z_sensitivity=float(yaw_speed),
                sim_device="cpu",
            )
        )
        self.gamepad.add_callback(carb.input.GamepadInput.X, self.reset)

    def reset(self) -> None:
        self.xy = self.initial_xy.clone()
        self.yaw = self.initial_yaw

    def close(self) -> None:
        del self.gamepad

    def update(self, dt: float) -> tuple[torch.Tensor, float]:
        command = self.gamepad.advance().cpu()
        self.xy[0] += float(dt) * float(command[0])
        self.xy[1] += float(dt) * float(command[1])
        self.xy[0] = torch.clamp(self.xy[0], min=self.x_min, max=self.x_max)
        self.xy[1] = torch.clamp(self.xy[1], min=self.y_min, max=self.y_max)
        self.yaw += float(dt) * float(command[2])
        return self.xy, self.yaw
