# HJB Regularization for Current PIRL Setup

This note documents the HJB term used in the current implementation.
It is a **soft critic regularizer** for PPO (not a standalone optimal-control solver).

## 1) Scope and role

Total training objective:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{ppo}
 + \mathcal{L}_{dyn}
 + \mathcal{L}_{hjb}
 \quad (\text{entropy/value terms are inside } \mathcal{L}_{ppo})
$$

HJB is used to shape critic geometry so that local value gradients stay consistent
with path-tracking error dynamics.

## 2) HJB state and control in current residual

Current HJB state is reduced to:

$$
x = \begin{bmatrix} d \\ \psi \end{bmatrix}
$$

where:

- $d$: signed cross-track error (meters),
- $\psi$: heading error to lookahead heading target (radians, wrapped to $[-\pi,\pi]$).

Control in the PDE residual is configurable by mode:

- `hjb_hamiltonian_mode = "policy"`: use current policy control \(u_\pi=[v,\omega]\),
- `hjb_hamiltonian_mode = "optimal"`: use analytic minimizer \(u^*\) from \(\partial \mathcal H/\partial u=0\).

For the optimal mode:

$$
u^*(x,\nabla V)=
\begin{bmatrix}
v^* \\
\omega^*
\end{bmatrix},
\quad
\frac{\partial \mathcal H}{\partial u}=0
$$

## 3) Error-state dynamics used in residual

Small-curvature local approximation:

$$
\dot d = v\sin\psi, \qquad \dot\psi = \omega
$$

This is intentionally simple and stable for regularization.
It matches the signed-error convention used by the environment.

## 4) Hamiltonian residual (reward-max convention)

PPO learns a reward-maximization value $V_\pi(x)=\mathbb E\left[\sum_t \gamma^t r_t\right]$,
so the HJB residual is written in the matching reward-max continuous form with
$r=-\ell$ and discount rate $\rho=-\ln\gamma/\Delta t$:

$$
\mathcal R_r(x,u,\nabla V)
=
r(x,u) + \nabla_x V(x)^\top f(x,u) - \rho\, V(x),
\qquad r = -\ell.
$$

At Bellman stationarity $\mathcal R_r(x,u^*)=0$, so squared-residual regularization
pulls $V$ toward a geometry consistent with the dynamics and the running-cost
model — without flipping gradients against PPO's TD targets.

Dynamics:

$$
f(x,u)=
\begin{bmatrix}\dot d\\\dot\psi\end{bmatrix}
=
\begin{bmatrix}v\sin\psi\\\omega\end{bmatrix}
$$

Running cost (shape of the reward prior, used via $r=-\ell$):

$$
\ell
=
w_t
 + w_d d^2
 + w_\psi (1-\cos\psi)
 + w_u\left(v^2 + 0.1\,\omega^2\right)
 - w_p \, v\cos\psi
$$

Interpretation:

- $w_d d^2$: penalize lateral deviation (matches env `cte_val = -d^2 * rew_scale_path_error`),
- $w_\psi(1-\cos\psi)$: penalize heading misalignment smoothly,
- $w_u(v^2+0.1\omega^2)$: regularize aggressive commands,
- $-w_p v\cos\psi$: reward forward progress toward the heading target.

For the reward-max Hamiltonian $\mathcal H_r(u)=-\ell+\nabla V^\top f$ (quadratic in $u$),
$\partial\mathcal H_r/\partial u=0$ gives the analytic maximizer:

$$
v^* = \frac{w_p\cos\psi + \frac{\partial V}{\partial d}\sin\psi}{2w_u},
\qquad
\omega^* = \frac{1}{0.2\, w_u}\frac{\partial V}{\partial \psi}
$$

Note the **positive** signs on $\partial V/\partial d$ and $\partial V/\partial\psi$:
these flipped when we switched from the cost-min to the reward-max convention.

Residual loss (optimal mode):

$$
\mathcal{L}_{hjb}
=
\lambda_{hjb}\,\mathbb{E}\!\left[\mathcal R_r(x,u^*,\nabla V)^2\right]
$$

## 5) Input indices in current `vec`

Current observation `vec` starts with:

$$
[\text{dot},\,\text{cross},\,v_x,\,v_y,\,\omega_z,\,d,\,\psi,\,\dots]
$$

Therefore:

- `hjb_vec_d_index = 5`
- `hjb_vec_psi_index = 6`

## 6) Raw vs normalized scale (important)

In implementation:

1. Critic forward pass inside HJB uses normalized state (`RunningStandardScaler`) so
   value prediction remains numerically stable.
2. HJB coordinates $d,\psi$ are taken from **raw vec** (physical units), and
   $\nabla V/\partial d,\nabla V/\partial\psi$ is computed w.r.t. those raw coordinates.

So HJB residual is evaluated in physically meaningful units while staying compatible
with normalized critic input.

## 7) Mapping to config keys

Main HJB config in `skrl_ppo_aux_cfg.yaml`:

- `hjb_loss_scale` $\rightarrow \lambda_{hjb}$
- `hjb_time_weight` $\rightarrow w_t$
- `hjb_distance_weight` $\rightarrow w_d$
- `hjb_heading_weight` $\rightarrow w_\psi$
- `hjb_control_weight` $\rightarrow w_u$
- `hjb_progress_weight` $\rightarrow w_p$
- `hjb_hamiltonian_mode` $\rightarrow$ choice of $u$ in Hamiltonian (`policy` or `optimal`)
- `hjb_step_dt` $\rightarrow \Delta t$ used to derive $\rho=-\ln\gamma/\Delta t$
- `hjb_max_lin_vel`, `hjb_max_ang_vel` are kept in config for backward compatibility.

The value used in the $\rho V$ term is the physical (unnormalized) critic output,
obtained by inverse-applying the `RunningStandardScaler` value preprocessor so
$\rho V$ has the same units as the reward-rate running cost $\ell$.

## 8) Practical expectations

- HJB loss can oscillate during PPO training because value and state distributions
  are non-stationary.
- This is normal if task metrics (progress, success, smoothness) improve.
- Use HJB as a regularizer (typically small-to-moderate scale), not as a dominant loss.

## 9) Notes on sign convention

Signed cross-track error is computed from the 2D cross product:

$$
\mathrm{sign}(d) = \mathrm{sign}\!\left(\hat t_x e_y - \hat t_y e_x\right),
\quad e = x_{robot} - x_{proj}
$$

If sign convention is flipped in a future refactor, both:

1. $d$ extraction in the environment, and
2. $\dot d$ model in HJB

must be updated consistently.
