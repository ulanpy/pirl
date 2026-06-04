# ObservationSchemaV2.1 Cookbook для ROS2/Nav2 деплоя

Этот документ фиксирует актуальный контракт между Isaac training и физическим ROS2 контроллером.
По сравнению с V2 в `vec` добавлен блок из `K=16` секторных LiDAR-хитов в body frame между
path window и memory tail. Этот блок используется как HJB-состоянием в критике, так и самой
политикой как дополнительные геометрические признаки.

## ONNX Inputs/Outputs

ONNX policy принимает:

```text
vec:       float32[1, 68]
costmap:   float32[1, 6, 100, 100]
rnn_state: float32[1, 1, 256]
```

ONNX policy возвращает:

```text
mean:          float32[1, 2]        # normalized [linear, yaw]
rnn_state_out: float32[1, 1, 256]
```

`scripts/toOnnx.py` встраивает `RunningStandardScaler` из SKRL checkpoint внутрь ONNX. На C++ стороне не нужно
применять `mean/std` scaler вручную. Нужно подать данные в формате ниже.

## Типичные ROS2 источники

- Odometry: `/odom` (`nav_msgs/Odometry`) или эквивалентный state estimator.
- TF: `map/odom -> base_link` для трансформации пути в frame робота.
- Path: `nav_msgs/Path` из Nav2 planner/controller pipeline.
- Local costmap: Nav2 local costmap data, обычно `nav2_msgs/Costmap` или `nav_msgs/OccupancyGrid`-подобный wrapper,
  в зависимости от интеграции.
- LiDAR scan: `sensor_msgs/LaserScan` от лазерного дальномера в `base_link` (или эквивалентном sensor frame).
  Используется для построения 16 секторных хитов в `vec`.
- Last action: последнее действие, отправленное ONNX/controller pipeline.
- Rewarder block: локальный блок, который считает те же reward components, что использовались при training.

## Runtime State

На reset нового goal, localization reset, emergency stop recovery или старте контроллера:

```text
rnn_state = zeros[1, 1, 256]
prev_action = zeros[2]
prev_reward_components = zeros[6]
costmap_history = unknown frames
```

На каждом control tick:

1. Собрать `costmap[6,100,100]`.
2. Собрать `vec[68]` (включая 16 секторных LiDAR-хитов).
3. Вызвать ONNX.
4. Сохранить `rnn_state_out` как следующий `rnn_state`.
5. Сохранить `mean` как `prev_action` для следующего tick.
6. Посчитать reward components и сохранить как `prev_reward_components`.

## Costmap Tensor

Форма:

```text
costmap = float32[6, 100, 100]
```

Физический размер:

```text
rolling window: 5.0 m x 5.0 m
resolution:     0.05 m/cell
grid:           100 x 100
history:        3 frames
```

Каждый history frame кодируется двумя каналами:

```text
cost       = 0.0 if unknown, otherwise nav2_cost / 254.0
known_mask = 0.0 if unknown, otherwise 1.0
```

Порядок каналов:

```text
0: newest_cost
1: newest_known_mask
2: previous_cost
3: previous_known_mask
4: oldest_cost
5: oldest_known_mask
```

Nav2 cost values:

```text
0       free
1..252  inflated/graded cost
253     inscribed
254     lethal obstacle
255     unknown
```

Пример C++-style логики:

```cpp
float cost_channel(uint8_t nav2_cost) {
  if (nav2_cost == 255) {
    return 0.0f;
  }
  return static_cast<float>(nav2_cost) / 254.0f;
}

float known_mask(uint8_t nav2_cost) {
  return nav2_cost == 255 ? 0.0f : 1.0f;
}
```

На reset history заполняется unknown, то есть все `cost=0`, все `known_mask=0`.

## Vec Tensor

Форма:

```text
vec = float32[68]
```

Layout:

```text
0      vx_body_mps
1      wz_body_radps
2      d_signed_m
3      heading_error_rad
4-27   path_window_base_link: 12 points * [x_m, y_m]
28-59  lidar_sectors_base_link: 16 sectors * [x_m, y_m]
60-61  prev_action: normalized [linear, yaw]
62-67  prev_reward_components: [progress, path_error, heading, proximity, collision, reverse]
```

### 0-1: Ego Velocity

Use robot body-frame velocity:

```text
vec[0] = vx in base_link, m/s
vec[1] = yaw rate wz, rad/s
```

For `nav_msgs/Odometry`, if twist is not already in `base_link`, rotate linear velocity into `base_link`.
For a tracked differential-drive robot, `vy` is intentionally not part of the contract.

### 2-3: Tracking Error

Use the same path adapter that builds the local path window.

```text
vec[2] = signed cross-track error d, meters
vec[3] = heading error psi, radians
```

Recommended sign convention:

```text
d_signed = sign(cross(path_tangent, robot_position - projection)) * distance_to_path
heading_error = normalize_angle(robot_yaw - path_heading_or_lookahead_heading)
```

Keep this convention identical between rewarder and observation builder.

### 4-27: Path Window

Take `nav_msgs/Path`, prune consumed points, resample by arc length, transform samples into `base_link`.

Recommended constants:

```text
window_size = 12
spacing_m = 0.10
format = [x0, y0, x1, y1, ..., x11, y11]
```

Algorithm:

1. Transform global/local path points into a common planning frame.
2. Find the nearest/projection point on the unconsumed path.
3. Sample 12 points at `s_current + i * 0.10 m`.
4. Transform each sampled point into `base_link`.
5. Fill `vec[4 + 2*i] = x_i`, `vec[4 + 2*i + 1] = y_i`.

If the path ends before all 12 samples, repeat/clamp to the final path point.

### 28-59: LiDAR Sector Hits

`K = 16` равных секторов покрывают весь lidar FOV (sim уровень: -180° .. +180°). На каждом
тике для каждого сектора выбирается ближайший hit и его положение записывается в `base_link`:

```text
window_size = 16
format = [x0, y0, x1, y1, ..., x15, y15]
spacing = full_fov / K   # in sim it is 360° / 16 = 22.5° per sector
```

Algorithm на ROS2 стороне (исходные данные из `sensor_msgs/LaserScan`):

1. Compute the same sector boundaries as in training: `K = 16` равных по угловому
   размеру слайсов всего FOV. На реальном LiDAR FOV обычно `[scan.angle_min, scan.angle_max]`.
2. Для каждого сектора `k`:
   - Среди ray indices `i` с `bearing(i)` в этом секторе и `scan.range_min <= r[i] < scan.range_max`,
     выбрать `i*` с минимальным `r[i]`.
   - Если все ray-и в секторе вышли в `range_max` или `inf`, использовать `r = scan.range_max`
     и любой `bearing` внутри сектора (например, `bearing(i*)` для замыкающего ray).
   - Записать `vec[28 + 2*k]   = r * cos(bearing(i*))`,
            `vec[28 + 2*k + 1] = r * sin(bearing(i*))`.
3. Если LiDAR/transform недоступны, заполнить блок нулями (в этом случае HJB-расширение
   просто не вносит вклада в политику; сравнимо с reset).

`base_link` подразумевается тем же, что и для path window (тангаж/наклон LiDAR можно
игнорировать на плоском полу). Если LiDAR установлен с фиксированным offset/yaw относительно
`base_link`, перевычислить bearing и (x,y) с учётом этого offset.

Sim генерирует hits в sensor frame, который для текущей конфигурации совпадает с
`base_scan` (см. `pirl_env_cfg.lidar.prim_path = ".../base_scan"` и `OffsetCfg(0,0,0)`),
поэтому контракт это body frame. На реальном роботе придерживайтесь того же frame.

### 60-61: Previous Action

Use the previous ONNX output before scaling to physical command:

```text
vec[60] = prev_action_linear_normalized
vec[61] = prev_action_yaw_normalized
```

On reset:

```text
prev_action = [0.0, 0.0]
```

Physical command scaling:

```text
v_cmd = mean[0] * max_lin_vel   # currently 0.5 m/s
w_cmd = mean[1] * max_ang_vel   # currently 1.5 rad/s
```

### 62-67: Previous Reward Components

Use reward components from the previous tick:

```text
62 progress
63 path_error
64 heading
65 proximity
66 collision
67 reverse
```

Clamp values to `[-1, 1]` before inserting into `vec`, matching training.

On reset:

```text
prev_reward_components = [0, 0, 0, 0, 0, 0]
```

## Minimal Pseudocode

```cpp
struct PirlRuntimeState {
  float rnn_state[1][1][256] = {};
  float prev_action[2] = {};
  float prev_reward_components[6] = {};
  CostmapHistory history;
};

Observation build_observation(
    const nav_msgs::msg::Odometry& odom,
    const nav_msgs::msg::Path& path,
    const Nav2Costmap& local_costmap,
    const sensor_msgs::msg::LaserScan& scan,
    PirlRuntimeState& state) {
  Observation obs;

  obs.costmap = encode_costmap_history(local_costmap, state.history);

  PathAdapterResult path_result = adapt_path_to_base_link(path);
  obs.vec[0] = body_vx(odom);
  obs.vec[1] = body_wz(odom);
  obs.vec[2] = path_result.d_signed;
  obs.vec[3] = path_result.heading_error;

  for (int i = 0; i < 12; ++i) {
    obs.vec[4 + 2 * i] = path_result.window[i].x;
    obs.vec[4 + 2 * i + 1] = path_result.window[i].y;
  }

  // 16 LiDAR sector hits in base_link.
  std::array<std::pair<float, float>, 16> sectors = encode_lidar_sectors(scan, /*K=*/16);
  for (int k = 0; k < 16; ++k) {
    obs.vec[28 + 2 * k] = sectors[k].first;
    obs.vec[28 + 2 * k + 1] = sectors[k].second;
  }

  obs.vec[60] = state.prev_action[0];
  obs.vec[61] = state.prev_action[1];
  for (int i = 0; i < 6; ++i) {
    obs.vec[62 + i] = clamp(state.prev_reward_components[i], -1.0f, 1.0f);
  }

  return obs;
}
```

## ONNX Inference Loop

```text
inputs:
  vec
  costmap
  rnn_state

outputs:
  mean
  rnn_state_out
```

Loop:

```cpp
auto obs = build_observation(odom, path, local_costmap, state);
auto [mean, next_rnn_state] = policy_onnx.run(obs.vec, obs.costmap, state.rnn_state);

state.rnn_state = next_rnn_state;
state.prev_action[0] = mean[0];
state.prev_action[1] = mean[1];

cmd_vel.linear.x = mean[0] * 0.5f;
cmd_vel.angular.z = mean[1] * 1.5f;
```

## Checklist

- `vec` shape is exactly `[1, 68]`.
- `costmap` shape is exactly `[1, 6, 100, 100]`.
- Costmap channel order is `[cost, known_mask]` for newest, previous, oldest frames.
- Path window is 12 points in `base_link`, resampled at `0.10 m`.
- LiDAR sectors are 16 `(x, y)` pairs in `base_link` covering the full LiDAR FOV with equal angular sectors.
- `d_signed` and `heading_error` use the same path adapter as the rewarder.
- `prev_action` is the previous normalized ONNX action, not physical `cmd_vel`.
- `prev_reward_components` are clipped to `[-1, 1]`.
- `rnn_state` is preserved between ticks and reset on new goal/recovery/localization reset.
- Do not apply SKRL `RunningStandardScaler` in C++; it is embedded in the exported ONNX by default.
