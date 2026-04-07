# PIRA Data Collection Plan (for teammate)

This document is an execution-ready plan for collecting real-robot data to train `f_real` for HJB regularization.

## 1) Goal of data collection

Collect transitions for one-step dynamics learning:

$$
(\mathbf{s}_t, \mathbf{u}_t, \Delta t_t) \rightarrow \mathbf{s}_{t+1}
$$

or equivalently:

$$
\Delta \mathbf{s}_t = \mathbf{s}_{t+1} - \mathbf{s}_t
$$

Primary target for first version:

$$
\mathbf{s}_{hjb} = [x_{rel}, y_{rel}]^T
$$

Optional expansion later:

$$
\mathbf{s}_{hjb} = [x_{rel}, y_{rel}, v_x, \omega_z]^T
$$

## 2) Core principles (must follow)

- Keep raw signals. Do not rely only on online computed features.
- Preserve temporal pairing `t -> t+1` (sequence index is required).
- Store real `dt` from timestamps, not a fixed assumed value.
- Mark events when goal/waypoint changes.
- Randomize scenarios across episodes (avoid one fixed route).

## 3) What to record every sample

At 20-50 Hz, record:

1. Time
- `t_ns` (monotonic or ROS time in nanoseconds)
- `dt` can be recomputed later from timestamps

2. Robot state (raw)
- `robot_x_map`, `robot_y_map`, `robot_yaw_map`
- `v_x_body`, `v_y_body` (if available), `w_z_body`

3. Control command
- `v_cmd` (`cmd_vel.linear.x`)
- `w_cmd` (`cmd_vel.angular.z`)

4. Goal/path info (raw)
- Coordinates of path points in map frame (at least active point):
  - `goal0_x_map`, `goal0_y_map`
  - optionally `goal1..goal3`
- `active_goal_index`
- `goal_switch_flag` (0/1)

5. Optional diagnostics
- wheel angular speeds (`omega_l`, `omega_r`)
- slip flag / controller mode

## 4) Why raw map poses are stored

Even if online code computes `x_rel, y_rel`, still store raw pose+goal.
Then compute relative quantities offline:

$$
\mathbf{p}_{rel}^{map} =
\begin{bmatrix}
g_x - x \\
g_y - y
\end{bmatrix}
$$

Rotate into body frame:

$$
\mathbf{p}_{rel}^{body} = R(-\psi)\,\mathbf{p}_{rel}^{map}
$$

where $\psi$ is robot yaw in map frame.

This avoids irreversible online mistakes and allows reprocessing.

## 5) Episode protocol

One episode = one short run (for example 20-60 s).

Within episode:
- Keep path definition stable.
- Robot moves toward active waypoint.
- If waypoint reached, switch to next and set `goal_switch_flag=1`.

Between episodes:
- Re-sample start pose and path geometry.
- Change route shape and distances.

Important:
- Goal switching during an episode is allowed, but must be explicitly marked.

## 6) How to place path points

Do not keep the same fixed points all day.

Recommended:
- Semi-random realistic placement in front/side sectors,
- different radii and turning profiles,
- avoid perfectly symmetric templates only.

Good default:
- 4 points per episode,
- randomized heading and curvature,
- keep physically traversable spacing.

## 7) Coverage checklist (must cover all)

Collect data with:
- straight motion (slow/fast),
- gentle turns,
- sharp turns,
- spin-in-place,
- stop/start transients,
- reverse (if used in deployment),
- different floor friction areas (if possible).

Without this, `f_real` overfits to narrow maneuvers.

## 8) Dataset format (CSV/Parquet schema)

Minimum columns:

- `episode_id`
- `step_id`
- `t_ns`
- `robot_x_map`, `robot_y_map`, `robot_yaw_map`
- `v_x_body`, `w_z_body` (plus `v_y_body` if available)
- `v_cmd`, `w_cmd`
- `goal0_x_map`, `goal0_y_map`
- `active_goal_index`
- `goal_switch_flag`

Recommended derived columns (offline):
- `dt`
- `x_rel`, `y_rel`
- `x_rel_next`, `y_rel_next`
- `delta_x_rel`, `delta_y_rel`

## 9) Train target definition for first surrogate

Input:

$$
\mathbf{z}_t = [x_{rel,t}, y_{rel,t}, v_{x,t}, \omega_{z,t}, v_{cmd,t}, \omega_{cmd,t}]
$$

Output (option A, preferred):

$$
\Delta \mathbf{y}_t =
\begin{bmatrix}
\Delta x_{rel,t} \\
\Delta y_{rel,t}
\end{bmatrix}
$$

with:

$$
\Delta x_{rel,t} = x_{rel,t+1} - x_{rel,t}, \quad
\Delta y_{rel,t} = y_{rel,t+1} - y_{rel,t}
$$

Output (option B):

$$
\dot{\mathbf{y}}_t = \Delta \mathbf{y}_t / \Delta t_t
$$

## 10) Data quality gates before training

Reject/fix dataset if:
- missing timestamps or non-positive `dt`,
- frequent TF jumps/discontinuities,
- unmarked goal switches,
- too many duplicate static samples (robot not moving for long periods),
- unit mismatch (`deg` vs `rad`, map vs odom confusion).

Quick sanity plots:
- histogram of `dt`,
- `v_cmd` vs `v_x_body`,
- trajectories in map frame,
- `x_rel,y_rel` continuity around goal switch events.

## 11) Split strategy

Split by episodes, not random rows:
- train: 70%
- val: 15%
- test: 15%

This prevents leakage from near-identical neighboring samples.

## 12) Suggested day plan (tomorrow)

1. Setup (30-45 min)
- verify topics, TF tree, and clock consistency
- dry-run logger for 2-3 minutes

2. Main collection (60-120 min)
- 20-50 Hz logging
- 30-60 episodes with varied paths
- mark anomalous runs

3. Validation pass (30 min)
- run quality gates
- generate derived columns (`x_rel,y_rel,delta`)

4. Export dataset
- one clean table (CSV or Parquet)
- one metadata file (robot config, units, date, floor type)

## 13) Final rule of thumb

For first stable PIRA run:
- use one active target point for HJB (`x_rel,y_rel`),
- keep model simple,
- prioritize clean timestamps and correctly marked goal switches over dataset size.
