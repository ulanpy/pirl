# PPODynamicsAuxRNN: detailed guide

This document explains how the custom agent `PPODynamicsAuxRNN` works in this project, how it differs from standard PPO in `skrl`, and how to run / debug it.

## Current PPO-RNN architecture (project snapshot)

The current setup is a recurrent PPO agent with an auxiliary dynamics objective:

- **Agent class**: `PPODynamicsAuxRNN` (inherits `skrl` `PPO_RNN`)
- **Model class**: `RecurrentSharedActorCritic` used for both policy and value (single shared instance)
- **Backbone**:
  - `vec` MLP: `15 -> 64 -> 64` (ELU)
  - `costmap` CNN: `4` stacked frames -> conv stack -> flattened features
  - feature fusion: `concat(vec_feat, cnn_feat) -> 256 -> 128` (ELU)
  - recurrent neck: `GRU(input=128, hidden=256, layers=1, batch_first=True)`
  - stabilization: `LayerNorm` before GRU and on GRU outputs
- **Heads**:
  - policy head: linear `256 -> 2` (normalized actions in `[-1, 1]`)
  - value head: linear `256 -> 1`
  - stochastic policy uses trainable `log_std_parameter`
- **RNN training window**:
  - sequence-aware mini-batching via `PPO_RNN`
  - default `sequence_length` from config: `64`
  - done-masked hidden-state reset inside recurrent forward pass
- **Aux dynamics head** (separate MLP in agent, not in backbone):
  - input: `[vec_t, action_t]`
  - target: `delta(vec) = vec_{t+1}[:N] - vec_t[:N]`
  - default `N = 5`
  - by default uses **normalized vec slices** (`dynamics_use_normalized_vec: True`)
- **Optimization objective**:
  - `policy_loss + value_loss + entropy_loss + dynamics_loss`
  - with gradient carrier from policy mean action into aux branch (so dynamics can shape policy/backbone)

In short: this is not "PPO + RNN wrapper only". It is a shared recurrent actor-critic with a GRU neck, sequence-aware PPO training, and an auxiliary dynamics objective that backpropagates into policy features.

## Short answer: is PPO implemented "from scratch"?

No. It is **not** a fully independent PPO implementation.

`PPODynamicsAuxRNN`:
- inherits from `skrl.agents.torch.ppo.ppo_rnn.PPO_RNN`
- reuses core PPO agent machinery from `skrl` (memory interface, preprocessors, optimizer/scaler handling, scheduler hooks, model API)
- overrides only parts needed for auxiliary dynamics learning:
  - custom config merge
  - storage of `next_states` for dynamics supervision
  - `_update()` loop with extra `dynamics_loss`

So this is best described as:
**"skrl PPO with a custom training update that adds auxiliary dynamics loss."**

---

## Files and responsibilities

- `ppo_dynamics_aux.py`
  - defines `PPODynamicsAux` class
  - defines `PPODynamicsAux_default_config`
  - overrides `init()` and `_update()`
- `runner_utils.py`
  - custom runner factory used by train/play
  - resolves `agent.class: PPODynamicsAux`
  - resolves `ppodynamicsaux_default_config`
  - instantiates custom agent with merged config
- `skrl_ppo_aux_cfg.yaml`
  - experiment config for this custom agent
  - sets `agent.class: PPODynamicsAux`
  - sets PPO hyperparameters and aux hyperparameters

---

## Config flow (important)

Configuration is merged in layers:

1) Base PPO defaults from `skrl`:
- `PPO_DEFAULT_CONFIG`

2) Aux-specific defaults:
- `PPODynamicsAux_default_config`
- keys:
  - `dynamics_loss_scale`
  - `dynamics_learning_rate`
  - `dynamics_hidden_layers`
  - `dynamics_target_dims`

3) YAML overrides (`agent:` block from `skrl_ppo_aux_cfg.yaml`)

Priority order:
**YAML > PPODynamicsAux defaults > skrl PPO defaults**

This means YAML values always win.

---

## Observation assumption

`PPODynamicsAux` assumes a **Dict observation space** with a key `"vec"`.

Why:
- the aux task predicts delta of compact vector state
- code infers where `"vec"` lives in flattened observation layout and slices it every batch

If `"vec"` is missing, initialization fails by design.

---

## Auxiliary task definition

For each sampled transition `(s_t, a_t, s_{t+1})`:

- extract:
  - `vec_t` from `s_t`
  - `vec_t+1` from `s_{t+1}`
- define target:
  - `delta_true = vec_t+1[:N] - vec_t[:N]`
  - where `N = dynamics_target_dims`
- model:
  - `dynamics_model([vec_t, action]) -> delta_pred`
- loss:
  - `dynamics_loss = dynamics_loss_scale * MSE(delta_pred, delta_true)`

This loss is added to the PPO objective in backprop.

---

## Update loop: what exactly happens

Inside `_update()`:

1) compute `returns` and `advantages` (GAE, same PPO style)
2) sample mini-batches from memory, including `next_states`
3) compute standard PPO losses:
   - policy surrogate loss
   - value loss
   - entropy loss (if enabled)
4) compute auxiliary dynamics loss
5) backpropagate:
   - `policy_loss + value_loss + entropy_loss + dynamics_loss`
6) optimizer step / scaler step / scheduler step
7) log metrics

So aux loss participates in the same backward pass as PPO losses.

---

## How dynamics loss reaches policy/backbone

Current implementation intentionally creates a differentiable path from dynamics objective to policy params.

Mechanism:
- forward side stays aligned with executed rollout action (`sampled_actions`)
- gradient side uses policy mean action as carrier

Implemented as:
- `dyn_actions = sampled_actions + (policy_mean_action - policy_mean_action.detach())`

Effect:
- forward value equals `sampled_actions`
- gradient flows through `policy_mean_action`
- therefore `dynamics_loss` can update policy/shared backbone parameters

When models are shared (`models.separate: False`), this also affects shared actor-critic features.

---

## Dynamics head architecture

`dynamics_model` is a small MLP:
- input: `[vec_t, action_t]`
- hidden: `dynamics_hidden_layers` (default `[128, 128]`)
- output: `dynamics_target_dims` (delta for first N vec components)

Learning rate:
- if `dynamics_learning_rate` is set -> use it
- else -> reuse PPO learning rate

Implementation detail:
- dynamics params are added as an extra param group to the same optimizer.

---

## TensorBoard signals to watch

Primary:
- `Loss / Policy loss`
- `Loss / Value loss`
- `Loss / Entropy loss` (if used)
- `Loss / Dynamics loss`

Aux influence diagnostics:
- `Grad / Dynamics-to-policy norm`
  - estimated norm of gradients of `dynamics_loss` wrt policy parameters
  - `> 0` means aux objective has a path to policy/backbone

Interpretation:
- very small but non-zero values can still be valid
- the useful quantity is often ratio vs PPO gradient scale and behavioral impact in A/B runs

---

## Runner integration

`get_runner(...)` in `runner_utils.py`:
- uses standard `skrl` path for built-in agents (`PPO`, `SAC`, etc.)
- for `PPODynamicsAux`:
  - resolves custom class
  - resolves custom default config
  - builds memory
  - merges config
  - constructs agent instance

So train/play stay generic while still supporting custom agent classes.

---

## How to run

Example training with aux config:

```bash
/isaac-sim/python.sh scripts/skrl/train.py --task jettank --agent skrl_ppo_aux_cfg_entry_point
```

Baseline run (no aux):
- use your baseline config entry point (the one mapped to standard PPO in environment registration)

Recommended experimental setup:
- same seeds
- baseline vs aux
- compare:
  - success/collision behavior
  - reward curves
  - convergence speed
  - dynamics and grad diagnostics

---

## Common pitfalls

1) Missing `"vec"` observation key
- aux agent depends on it

2) Too small `dynamics_loss_scale`
- aux signal exists but has little practical impact

3) Too large `dynamics_loss_scale`
- can destabilize PPO optimization

4) Misreading losses by magnitude only
- compare trends and behavior, not raw scales across different objectives

5) Assuming aux head alone guarantees obstacle avoidance
- this aux currently predicts ego-dynamics deltas, not explicit collision risk

---

## Practical tuning tips

Start conservative:
- `dynamics_loss_scale`: `0.02 -> 0.05`
- keep PPO lr unchanged first

Then:
- monitor `Grad / Dynamics-to-policy norm`
- run A/B with `scale=0` vs `scale>0`
- increase to `0.1` only if training remains stable and aux impact is still weak

If instability appears:
- reduce `dynamics_loss_scale`
- reduce `dynamics_learning_rate`
- reduce `dynamics_target_dims`

---

## Current design limits

- Aux supervision target is only `delta vec` for first N components
- This improves motion modeling but is not a direct safety classifier
- For stronger obstacle-avoidance effect, consider extending aux targets:
  - short-horizon collision risk
  - time-to-collision proxies
  - occupancy change prediction in local map

---

## Minimal mental model

Think of this agent as:
- PPO that still optimizes return
- plus an extra self-supervised dynamics objective
- with gradient routing so the aux objective can shape policy features

That is exactly the intended PINN-inspired bridge in this implementation.

