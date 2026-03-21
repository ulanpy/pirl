# HJB for local planner (time + distance)

This note rewrites the same idea in clean LaTeX form.

## 1) System and objective

State and control:

$$
\mathbf{s} = \begin{bmatrix}x \\ y \\ \theta\end{bmatrix}, \qquad
\mathbf{u} = \begin{bmatrix}v \\ \omega\end{bmatrix}
$$

Nonholonomic (unicycle) kinematics:

$$
\dot{x} = v\cos\theta, \qquad
\dot{y} = v\sin\theta, \qquad
\dot{\theta} = \omega
$$

Compactly:

$$
\dot{\mathbf{s}} = f(\mathbf{s}, \mathbf{u})
$$

Goal: reach $(x_g, y_g)$ quickly with feasible/smooth control.

## 2) Continuous-time cost (time + distance)

Let $\mathbf{p} = [x, y]^T$ and $\mathbf{p}_g = [x_g, y_g]^T$.

$$
J(\mathbf{s}_0)
= \int_{0}^{T}
\left(
w_t
 + w_d \|\mathbf{p}(t) - \mathbf{p}_g\|_2^2
 + w_v v(t)^2
 + w_\omega \omega(t)^2
\right)\,dt
$$

where:
- $w_t>0$ encourages minimum-time behavior
- $w_d>0$ encourages goal approach
- $w_v, w_\omega>0$ regularize control effort.

## 3) Value function and HJB equation

Value function:

$$
V(\mathbf{s}) = \min_{\mathbf{u}(\cdot)} J(\mathbf{s})
$$

Running cost:

$$
\ell(\mathbf{s}, \mathbf{u})
=
w_t + w_d \|\mathbf{p}-\mathbf{p}_g\|_2^2 + w_v v^2 + w_\omega \omega^2
$$

HJB (infinite-horizon stationary form):

$$
0 = \min_{\mathbf{u}}
\left[
\ell(\mathbf{s}, \mathbf{u})
 + \nabla_{\mathbf{s}}V(\mathbf{s})^T f(\mathbf{s}, \mathbf{u})
\right]
$$

Define Hamiltonian:

$$
\mathcal{H}(\mathbf{s}, \mathbf{u}, \nabla V)
=
\ell(\mathbf{s}, \mathbf{u})
 + \nabla_{\mathbf{s}}V(\mathbf{s})^T f(\mathbf{s}, \mathbf{u})
$$

Then optimality is:

$$
\min_{\mathbf{u}} \mathcal{H} = 0
$$

## 4) Why $\nabla V^T f$ appears (chain rule)

Along any trajectory $\mathbf{s}(t)$:

$$
\frac{d}{dt}V(\mathbf{s}(t))
=
\nabla_{\mathbf{s}}V(\mathbf{s})^T \dot{\mathbf{s}}
=
\nabla_{\mathbf{s}}V(\mathbf{s})^T f(\mathbf{s}, \mathbf{u})
$$

Interpretation:
- $\nabla V$ is sensitivity of future cost to state coordinates.
- Multiplying by $f$ gives instantaneous change of value under control $\mathbf{u}$.
- HJB balances instantaneous running cost and expected change in future cost.

## 5) Local-frame surrogate (practical RL form)

Instead of global $(x,y,\theta)$, use local goal-relative state:

$$
\mathbf{s}_{rel} =
\begin{bmatrix}
x_{rel} \\
y_{rel}
\end{bmatrix}
\quad (\text{optionally } \theta_{rel})
$$

Common local relative dynamics approximation:

$$
\dot{x}_{rel} = -v + \omega y_{rel}, \qquad
\dot{y}_{rel} = -\omega x_{rel}
$$

Local Hamiltonian residual:

$$
r_{hjb}
=
\ell_{rel}(\mathbf{s}_{rel}, \mathbf{u})
 + \frac{\partial V}{\partial x_{rel}} \dot{x}_{rel}
 + \frac{\partial V}{\partial y_{rel}} \dot{y}_{rel}
$$

Loss:

$$
\mathcal{L}_{hjb}
=
\lambda_{hjb}\,\mathbb{E}\left[r_{hjb}^2\right]
$$

This is the "local HJB surrogate": a practical regularizer, not a full global PDE on map coordinates.

## 6) Replace analytic dynamics with learned real surrogate

If real-data surrogate is trained:

$$
\dot{\mathbf{s}} \approx f_{real}(\mathbf{s}, \mathbf{u})
$$

or in discrete form:

$$
\mathbf{s}_{t+1}
\approx
\mathbf{s}_t + \Delta t\, f_{real}(\mathbf{s}_t, \mathbf{u}_t)
$$

then HJB residual becomes:

$$
r_{hjb}
=
\ell(\mathbf{s}, \mathbf{u})
 + \nabla_{\mathbf{s}}V(\mathbf{s})^T f_{real}(\mathbf{s}, \mathbf{u})
$$

and:

$$
\mathcal{L}_{hjb}
=
\lambda_{hjb}\,\mathbb{E}\left[r_{hjb}^2\right]
$$

This is exactly the PIRA anchor: critic geometry is pushed toward real robot dynamics.

## 7) Discrete training recipe (batch)

For each minibatch:

1. Compute $V(\mathbf{s})$ from critic.
2. Compute $\nabla_{\mathbf{s}}V$ with autograd.
3. Compute $f(\mathbf{s},\mathbf{u})$ (analytic or $f_{real}$).
4. Build $\ell(\mathbf{s},\mathbf{u})$.
5. Build residual:

$$
r = \ell + \sum_i \frac{\partial V}{\partial s_i} f_i
$$

6. Add:

$$
\mathcal{L}_{hjb} = \lambda\,\mathrm{mean}(r^2)
$$

## 8) Meaning of $x_{rel}, y_{rel}$ in your current env

In your current setup, the observation does **not** pass absolute map/odom robot coordinates.
It passes local path-segment points transformed into robot frame and flattened into `vec`.

So in your case:
- $x_{rel}, y_{rel}$ are target/path-point coordinates relative to robot body frame
- not absolute robot pose in map/odom.

With your config:
- `hjb_vec_x_index: 5`
- `hjb_vec_y_index: 6`

these indices refer to the first local path point $(x_{rel}, y_{rel})$ in `vec`.

## 9) Recommended first stable PIRA-HJB setup

Use a minimal state for HJB branch:

$$
\mathbf{s}_{hjb} = [x_{rel}, y_{rel}]^T,
\qquad
\mathbf{u} = [v_{cmd}, \omega_{cmd}]^T
$$

Start with simple running cost:

$$
\ell
=
w_t
 + w_d \sqrt{x_{rel}^2 + y_{rel}^2 + \varepsilon}
 + w_u\left(v^2 + c_\omega \omega^2\right)
$$

Then swap analytic $f$ with trained $f_{real}$ when surrogate is ready.

## 10) Q&A (practical decisions)

### Q1. Must $\mathbf{s}_{hjb}$ always be a subset of the policy state?

Short answer: in your current implementation, **yes in practice** (recommended and simplest).

Why:
- the critic already receives the policy state (or its preprocessed form),
- HJB gradient is computed as $\nabla V$ with respect to that critic input tensor,
- indices such as `hjb_vec_x_index`, `hjb_vec_y_index` select coordinates from that same input.

So the clean setup is:

$$
\mathbf{s}_{hjb} = P\mathbf{s}_{policy}
$$

where $P$ is a coordinate-selection/projection matrix.

Could $\mathbf{s}_{hjb}$ be not a subset? Theoretically yes, if you redesign critic/HJB branch to explicitly take extra variables and differentiate w.r.t. them. But this is more complex and usually not needed for first working version.

### Q2. Why do we have freedom to choose only a slice for HJB?

Because HJB here is an **auxiliary regularizer**, not the full exact PDE over all observation channels.

Total optimization remains:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{ppo}
 + \mathcal{L}_{hjb}
\quad
(\text{plus optional auxiliary terms})
$$

and

$$
\mathcal{L}_{hjb}
=
\lambda_{hjb}\,\mathbb{E}\!\left[
\left(
\ell(\mathbf{s}_{hjb},\mathbf{u})
 + \nabla_{\mathbf{s}_{hjb}}V \cdot f(\mathbf{s}_{hjb},\mathbf{u})
\right)^2
\right]
$$

As long as $\mathbf{s}_{hjb}$ is physically meaningful and $f$ is reliable on it, the regularizer is valid and useful.

### Q3. How does adding/removing elements in $\mathbf{s}_{hjb}$ change behavior?

If coordinate $z$ is included in $\mathbf{s}_{hjb}$, HJB shapes $\partial V/\partial z$.
If $z$ is excluded, HJB does not directly constrain that direction.

Formally, with:

$$
\mathbf{s}_{hjb}=[s_1,\dots,s_k]^T,\qquad
r=\ell+\sum_{i=1}^{k}\frac{\partial V}{\partial s_i}f_i
$$

only $(s_1,\dots,s_k)$ contribute to residual $r$.

### Q4. What if the slice is too small?

Pros: stable, low-variance gradients, easy surrogate fitting.

Risk: weak physical anchoring; some behaviors remain unconstrained by HJB.

Typical symptom: good goal attraction but limited correction of turn/velocity realism.

### Q5. What if the slice is too large?

Pros: richer physical constraints in principle.

Risks (common):
- surrogate model error grows with dimension,
- HJB gradients become noisy/conflicting,
- optimization may destabilize PPO updates.

A practical bias-variance trade-off:

$$
\text{small }k \Rightarrow \text{low variance, higher bias},\qquad
\text{large }k \Rightarrow \text{lower bias, higher variance}
$$

### Q6. What should be included first for your setup?

Start with:

$$
\mathbf{s}_{hjb}^{(2D)}=[x_{rel},y_{rel}]
$$

Then expand only if stable:

$$
\mathbf{s}_{hjb}^{(4D)}=[x_{rel},y_{rel},v_x,\omega_z]
$$

Avoid putting purely historical channels (e.g., previous reward) into first HJB version unless you explicitly model their dynamics.
