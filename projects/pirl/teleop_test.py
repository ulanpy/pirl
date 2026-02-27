from isaaclab.app import AppLauncher
import argparse

# 1. Initialize AppLauncher (MUST be before any other Isaac imports)
parser = argparse.ArgumentParser(description="Open-loop test for Jettank.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Now we can import the rest
import torch
import time
from pirl.tasks.direct.pirl.pirl_env import PirlEnv
from pirl.tasks.direct.pirl.pirl_env_cfg import PirlEnvCfg


def run_turn_phase(env, cfg, turn_cmd: float, title: str) -> tuple[float, float]:
    steps = int(3.0 / (cfg.sim.dt * cfg.decimation))
    yaw_rates = []

    print(title)
    for i in range(steps):
        actions = torch.tensor([[0.0, turn_cmd]], device=env.device)
        obs, _, _, _, _ = env.step(actions)
        yaw_rate = obs["policy"]["vec"][0, 4].item()
        yaw_rates.append(yaw_rate)
        if i % 20 == 0:
            print(f"  Step {i:3d} | Yaw Rate: {yaw_rate:+.4f}")

    # Ignore transient startup and evaluate steady segment
    steady = yaw_rates[40:] if len(yaw_rates) > 40 else yaw_rates
    mean_abs = float(sum(abs(v) for v in steady) / max(len(steady), 1))
    std_abs = float(torch.tensor([abs(v) for v in steady]).std(unbiased=False).item()) if steady else 0.0
    return mean_abs, std_abs


def run_test():
    # Configure deterministic open-loop conditions
    cfg = PirlEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 100.0

    env = PirlEnv(cfg=cfg, render_mode=None)
    env.reset()

    print("\n--- STARTING OPEN-LOOP TEST ---")

    # Left test from reset
    env.reset()
    left_mean, left_std = run_turn_phase(env, cfg, turn_cmd=1.0, title="Testing LEFT turn (w=+0.5) for 3 seconds...")

    # Small settle + fresh reset before right test
    env.step(torch.tensor([[0.0, 0.0]], device=env.device))
    time.sleep(0.2)
    env.reset()
    right_mean, right_std = run_turn_phase(env, cfg, turn_cmd=-1.0, title="\nTesting RIGHT turn (w=-0.5) for 3 seconds...")

    asymmetry = abs(left_mean - right_mean) / max((left_mean + right_mean) * 0.5, 1e-8)
    print("\n--- SUMMARY (steady-state |yaw_rate|) ---")
    print(f"  LEFT  mean={left_mean:.4f}, std={left_std:.4f}")
    print(f"  RIGHT mean={right_mean:.4f}, std={right_std:.4f}")
    print(f"  Asymmetry ratio: {asymmetry * 100:.2f}%")

    print("\n--- TEST FINISHED ---")
    simulation_app.close()

if __name__ == "__main__":
    run_test()
