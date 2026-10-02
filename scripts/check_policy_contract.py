#!/usr/bin/env python3
"""Smoke-check the PIRL flattened observation layout against actor and value CNN inputs."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate the PIRL policy observation contract.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

from pirl.tasks.navigation.pirl.agents.recurrent_models import (
    FeedForwardDeterministicValue,
    RecurrentGaussianPolicy,
)


def main() -> None:
    observation_space = gym.spaces.Dict(
        {
            "costmap": gym.spaces.Box(low=0.0, high=1.0, shape=(1, 100, 100), dtype=float),
            "vec": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(34,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    policy = RecurrentGaussianPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        num_envs=1,
        sequence_length=64,
        transformer_model_dim=128,
        transformer_num_heads=4,
        transformer_num_layers=2,
        transformer_context_len=32,
        transformer_ff_dim=512,
        aux_dim=6,
    )
    value = FeedForwardDeterministicValue(
        observation_space=observation_space, action_space=action_space, device="cpu"
    )
    # skrl Dict flattening is sorted: costmap[10000], then vec[34].
    flat_state = torch.zeros((1, 10000 + 34))
    mean, extras = policy.compute({"states": flat_state})
    value_prediction, _ = value.compute({"states": flat_state})
    if tuple(mean.shape) != (1, 2) or tuple(value_prediction.shape) != (1, 1):
        raise AssertionError(f"Unexpected model outputs: mean={tuple(mean.shape)}, value={tuple(value_prediction.shape)}")
    if tuple(extras["rnn"][0].shape) != (32, 1, 129):
        raise AssertionError(f"Unexpected transformer cache: {tuple(extras['rnn'][0].shape)}")
    print("Policy contract OK: costmap=(1,100,100), vec=(34,), action=(2,)")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
