# PPODynamicsAuxRNN: Current Pipeline (with LaTeX)

This document describes the **current** implementation in this repository:

- PPO-RNN agent (`PPODynamicsAuxRNN`)
- recurrent actor (`RecurrentGaussianPolicy`)
- feed-forward critic (`FeedForwardDeterministicValue`)
- auxiliary dynamics head
- HJB PINN-style regularizer for the critic

## 1) High-level architecture

### Actor (recurrent)

- Input: flattened `Dict(obs)` with keys `vec` and `costmap`
- `vec` branch:
  - split into `core_vec` and optional `aux_vec` tail (`aux_dim`)
  - core MLP: `core_dim -> 64 -> 64`
  - aux MLP (if `aux_dim > 0`): `aux_dim -> 32 -> 32`
- `costmap` branch:
  - CNN (4 conv layers) -> flatten
- Fusion:
  - concat features -> MLP `-> 256 -> 128`
  - `LayerNorm` before GRU
- Recurrent neck:
  - `GRU(128 -> 256, num_layers=1, batch_first=True)`
  - output `LayerNorm`
- Output head:
  - mean action head `Linear(256 -> 2)`
  - trainable `log_std_parameter`

### Critic (feed-forward)

- No recurrent state
- `vec` MLP + `costmap` CNN -> concat -> fusion MLP -> `value_head (128 -> 1)`

## 2) Observation and action

Current `vec` size in env config:

$$
\text{vec\_dim} = 5 + 2 \cdot \text{path\_num\_points} + 2 + 7
$$

With `path_num_points = 4`:

$$
\text{vec\_dim} = 22
$$

`costmap` shape is `(4, 100, 100)`.

Action is normalized:

$$
a_t = [a_v, a_\omega] \in [-1, 1]^2
$$

Scaled control:

$$
v_t = a_v \cdot v_{\max}, \quad \omega_t = a_\omega \cdot \omega_{\max}
$$

with \(v_{\max}=0.5\), \(\omega_{\max}=3.0\).

## 3) PPO-RNN batching logic

Current config:

- `rollouts = 256`
- `num_envs = 10`
- `sequence_length = 64`
- `mini_batches = 4`

Transitions per update:

$$
N = 256 \cdot 10 = 2560
$$

Per mini-batch:

$$
N_{mb} = \frac{2560}{4} = 640
$$

Validity condition for RNN sequence batching:

$$
N_{mb} \bmod \text{sequence\_length} = 0
$$

Here \(640 \bmod 64 = 0\), so sequence chunks are valid.

## 4) Training objective

Total loss:

$$
\mathcal{L}_{total}
= \mathcal{L}_{ppo}
 + \mathcal{L}_{value}
 + \mathcal{L}_{entropy}
 + \mathcal{L}_{dyn}
 + \mathcal{L}_{hjb}
$$

### PPO surrogate

$$
r_t(\theta)=\exp\left(\log \pi_\theta(a_t|s_t)-\log \pi_{old}(a_t|s_t)\right)
$$

$$
\mathcal{L}_{ppo}
= - \mathbb{E}\left[
\min\left(r_t A_t,\;\text{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)
\right]
$$

### Value loss

$$
\mathcal{L}_{value}
= c_v \cdot \text{MSE}\left(V_\phi(s_t), \hat{R}_t\right)
$$

### Entropy loss

$$
\mathcal{L}_{entropy}
= -c_e \cdot \mathbb{E}\left[\mathcal{H}\big(\pi_\theta(\cdot|s_t)\big)\right]
$$

### Auxiliary dynamics loss

For first \(N_d=\text{dynamics\_target\_dims}\) vec components:

$$
\Delta \mathbf{v}^{true}_t
= \mathbf{v}_{t+1}^{(1:N_d)} - \mathbf{v}_t^{(1:N_d)}
$$

$$
\Delta \mathbf{v}^{pred}_t = f_{dyn}\left([\mathbf{v}_t, a_t^{dyn}]\right)
$$

$$
\mathcal{L}_{dyn}
= \lambda_{dyn}\cdot \text{MSE}\left(\Delta \mathbf{v}^{pred}_t,\Delta \mathbf{v}^{true}_t\right)
$$

Gradient-carrier action:

$$
a_t^{dyn}
= a_t + \left(\mu_\theta(s_t)-\operatorname{stopgrad}(\mu_\theta(s_t))\right)
$$

Forward value stays at rollout action \(a_t\), while gradients flow through policy mean \(\mu_\theta\).

### HJB PINN-style regularizer

The critic is differentiated by autograd w.r.t. critic input:

$$
\nabla_s V_\phi(s_t)
$$

Using relative-goal coordinates \((x_t, y_t)\) from `vec`:

$$
\dot{x}_t = -v_t + \omega_t y_t,\qquad
\dot{y}_t = -\omega_t x_t
$$

Running cost:

$$
\ell_t
= w_t + w_d\sqrt{x_t^2+y_t^2+\varepsilon}
 + w_u\left(v_t^2 + 0.1\omega_t^2\right)
$$

Hamiltonian residual:

$$
\mathcal{H}_t
= \ell_t
 + \frac{\partial V}{\partial x}\dot{x}_t
 + \frac{\partial V}{\partial y}\dot{y}_t
$$

HJB loss:

$$
\mathcal{L}_{hjb}
= \lambda_{hjb}\cdot \mathbb{E}\left[\mathcal{H}_t^2\right]
$$

## 5) Important runtime details

- HJB branch is computed in FP32 block for gradient stability.
- If critic-input gradient is disconnected for a mini-batch, the code uses a safe fallback
  (zero gradient tensor) instead of crashing.
- Done masks are used to reset GRU hidden states within sequence processing.

## 6) Key TensorBoard metrics

- `Loss / Policy loss`
- `Loss / Value loss`
- `Loss / Entropy loss`
- `Loss / Dynamics loss`
- `Loss / HJB loss`
- `Grad / Dynamics-to-policy norm`
- `Policy / Standard deviation`
- `Learning / Learning rate`

## 7) Run command

```bash
/isaac-sim/python.sh scripts/skrl/train.py --task jettank --agent skrl_ppo_aux_cfg_entry_point
```

