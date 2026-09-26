# Manager-Based PIRL Migration

## Objective

Move PIRL from a task-specific `DirectRLEnv` lifecycle to Isaac Lab's
`ManagerBasedRLEnv` while preserving the policy contract:

- action: normalized `[linear, yaw]`;
- observation: `{"policy": {"vec": [N, 34], "costmap": [N, 2, 100, 100]}}`;
- recurrent state remains entirely in the policy;
- reward and termination semantics stay explicit and independently configurable.

## Pinned Isaac Lab contract

The migration is based on the vendored Isaac Lab implementation, not a changing
external API:

- `ManagerBasedRLEnv.step()` processes actions, runs decimated physics, computes
  termination and reward terms, resets finished environments, then computes observations.
- An `ActionTerm` processes a policy action once per environment step and applies it
  once per physics step.
- `ObservationGroupCfg(concatenate_terms=False)` preserves PIRL's `vec` and
  `costmap` dictionary inputs.
- Reset `EventTerm`s receive only the reset environment IDs; this is the correct
  place to sample robot spawn, path, and obstacle scenarios.

Relevant local sources:

- `third_party/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
- `third_party/IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py`
- `third_party/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py`
- `third_party/IsaacLab/source/isaaclab/isaaclab/managers/manager_term_cfg.py`

## Target structure

```text
manager_based/
  env.py                  # small owner of scene assets and NavigationState
  env_cfg.py              # scene and manager-term composition
  mdp/
    actions.py            # [v, w] -> wheel velocity action term
    observations.py       # vec and current costmap terms
    rewards.py            # independent reward terms
    terminations.py       # collision, success, timeout
    events.py             # reset robot, path, and scenario
  navigation_state.py     # one cache shared by reward/done/observation terms
  scenario.py             # future static/dynamic task-family sampler
```

`LocalPathManager`, `LocalCostmapBuilder`, and `DynamicObstacles` remain stateful
services. Manager terms read their results; they do not duplicate their state.

## Migration stages

1. **Contract tests** — record action/observation shapes, resets, and reward
   component values for fixed seeds.
2. **Shared state** — refresh path projection,
   LiDAR ranges, final-goal distance, and current costmap once after physics. All
   MDP terms read this cache.
3. **Manager task** — add a manager-based task with declarative scene,
   custom wheel action term, non-concatenated policy observations, reward terms,
   termination terms, and reset events.
4. **Smoke test** — run zero/random action tests and a short PPO-RNN train in
   Isaac Lab; verify shapes, reset behavior, and stable rollout updates.
5. **Cutover (complete)** — `burger` is now the manager-based task; the legacy
   `DirectRLEnv` implementation and its unused USD obstacle helpers are removed.
6. **Future extension** — add curriculum terms that change scenario-generator
   parameters and privileged observation groups for teacher/IL experiments.

## Non-negotiable invariants

- Never let an observation, reward, and termination term recompute navigation state
  independently; that makes the same timestep internally inconsistent.
- Keep reward terms side-effect free except for one explicit reward-history update
  owned by `NavigationState`.
- Do not add observation history to the costmap: temporal information belongs to the
  recurrent policy cache.
- Keep a fixed pool of obstacle slots. Scenario generation changes active masks,
  poses, and motion-policy parameters only during reset.
