# Observation Space для деплоя на реального робота

Полная документация по формированию наблюдений при переносе обученной модели с Isaac Sim на физического робота.

---

## 📊 Общая структура наблюдения

```
observation = {
    "vec": [1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 24 + 2 + 6] = 41 элемент
    "costmap": [3 каналов, 100 пиксели, 100 пиксели]
}
```

Итого:
- **vec**: 41 скаляр
- **costmap**: 3 × 100 × 100 = 30,000 значений

После **флаттенинга в sorted порядке** (что делает skrl):
- costmap first (sorted alphabetically)
- vec second
- **Общий вектор**: 30,041 элемент → нормализуется через RunningStandardScaler

---

## 🎯 VEC Компоненты (41 элемент)

### Индекс 0-1: Направление команды (2 элемента)

| Индекс | Имя | Формула | Диапазон | Смысл |
|--------|-----|---------|----------|-------|
| **0** | `dot_cmd` | `forward · command` | [-0.5, 0.5] | Dot product с вектором пути |
| **1** | `cross_cmd` | `forward × command` | [-0.5, 0.5] | Cross product для боковой ошибки |

**Физика на роботе:**
```python
# Получить направление движения робота (unit vector в base_link)
robot_forward = [1, 0]  # X-axis в robot frame (всегда вперед)

# Получить направление пути (ближайшая следующая точка)
path_point = [px, py]  # в robot frame
path_direction = normalize([px, py])

# Dot product
dot_cmd = robot_forward[0] * path_direction[0] + robot_forward[1] * path_direction[1]
# Clamp в [-0.5, 0.5] (небольшой запас)
dot_cmd = clamp(dot_cmd, -0.5, 0.5)

# Cross product (только Z компонента в 2D)
cross_cmd = robot_forward[0] * path_direction[1] - robot_forward[1] * path_direction[0]
cross_cmd = clamp(cross_cmd, -0.5, 0.5)
```

---

### Индекс 2-4: Скорости в robot frame (3 элемента)

| Индекс | Имя | Источник | Диапазон | Единица |
|--------|-----|----------|----------|---------|
| **2** | `forward_speed` (vx) | IMU/Odometry | [-0.5, 0.5] | м/с |
| **3** | `lateral_speed` (vy) | IMU/Odometry | [-0.1, 0.1] | м/с |
| **4** | `yaw_rate` (wz) | IMU | [-1.5, 1.5] | рад/с |

**Физика на роботе:**
```python
# Трансформировать линейные скорости из world frame в robot frame
vx_world, vy_world = odometry.twist.linear.x, odometry.twist.linear.y
robot_yaw = tf.get_robot_yaw()

# Rotation matrix для мир → robot
cos_yaw = cos(robot_yaw)
sin_yaw = sin(robot_yaw)

vx_robot = cos_yaw * vx_world + sin_yaw * vy_world
vy_robot = -sin_yaw * vx_world + cos_yaw * vy_world
wz_robot = odometry.twist.angular.z

# Нормализация (может быть вне этих диапазонов, но обычно нет)
vx_robot = clamp(vx_robot, -0.5, 0.5)
vy_robot = clamp(vy_robot, -0.1, 0.1)
wz_robot = clamp(wz_robot, -1.5, 1.5)
```

**Примечание**: Forward направление всегда X-axis в robot frame (перед робота).

---

### Индекс 5-6: Ошибки отслеживания пути (2 элемента)

| Индекс | Имя | Тип | Диапазон | Единица |
|--------|-----|-----|----------|---------|
| **5** | `curr_path_error_signed` (d) | Signed cross-track error | [-2.5, 2.5] | метры |
| **6** | `heading_error` (ψ) | Heading mismatch | [-π, π] | радианы |

**Физика на роботе:**

```python
# d: Closest point на пути, найти перпендикулярное расстояние
# (Алгоритм: ближайшая точка пути)
closest_idx = find_closest_point_on_path(robot_pos)
closest_point = path[closest_idx]

# Вектор от точки пути к роботу (в world frame)
error_vec_w = robot_pos - closest_point

# Трансформировать в robot frame
error_vec_body = transform_to_robot_frame(error_vec_w, robot_yaw)

# Signed cross-track error (положительный справа от пути)
d = error_vec_body[1]  # Y-компонента
d = clamp(d, -2.5, 2.5)

# ψ: Heading error (разница между направлением робота и пути)
path_heading = atan2(path[closest_idx+1].y - closest_point.y,
                       path[closest_idx+1].x - closest_point.x)
robot_heading = atan2(sin(robot_yaw), cos(robot_yaw))

# Нормализовать угловую разницу в [-π, π]
psi = normalize_angle(robot_heading - path_heading)
```

---

### Индекс 7-8: Ближайшее препятствие от LiDAR (2 элемента)

| Индекс | Имя | Значение | Диапазон | Единица |
|--------|-----|----------|----------|---------|
| **7** | `nearest_obstacle_x` | X ближайшего луча | [-18, 18] | метры |
| **8** | `nearest_obstacle_y` | Y ближайшего луча | [-18, 18] | метры |

**Физика на роботе:**

```python
# Найти ближайший LiDAR луч
lidar_ranges = lidar_msg.ranges  # [num_rays] в метрах
lidar_angles = lidar_msg.angle_min + np.arange(len(lidar_ranges)) * lidar_msg.angle_increment

# Найти минимальный диапазон
min_idx = argmin(lidar_ranges[:num_rays])
min_range = lidar_ranges[min_idx]
min_angle = lidar_angles[min_idx]

# Конвертировать в robot frame (X вперед, Y влево)
nearest_obstacle_x = min_range * cos(min_angle)
nearest_obstacle_y = min_range * sin(min_angle)

# Clamp к максимальному диапазону скенера
max_lidar_range = 18.0  # метры
nearest_obstacle_x = clamp(nearest_obstacle_x, -max_lidar_range, max_lidar_range)
nearest_obstacle_y = clamp(nearest_obstacle_y, -max_lidar_range, max_lidar_range)
```

---

### Индекс 9-32: Локальное окно пути (24 элемента = 12 точек × 2)

| Раздел | Формат | Кол-во | Диапазон | Смысл |
|--------|--------|--------|----------|-------|
| **9-32** | [x₀, y₀, x₁, y₁, ..., x₁₁, y₁₁] | 12 точек | [-3, 3] | Последующие 12 точек пути |

**Конфигурация пути:**
- Длина пути: **6.0 м**
- Интервал между точками: **0.1 м**
- Всего точек на пути: 61 точка
- **Видимое окно**: 12 ближайших точек (1.2 м вперед)

**Физика на роботе:**

```python
# Получить глобальный путь
global_path = nav_plan.poses  # list of Pose

# Найти текущий индекс (ближайшая точка)
curr_idx = find_closest_point_on_path(robot_pos, global_path)

# Извлечь окно (12 точек)
window_size = 12
path_window = []
for i in range(window_size):
    idx = min(curr_idx + i, len(global_path) - 1)
    point = global_path[idx].position
    path_window.append([point.x, point.y])

# Трансформировать в robot frame
robot_yaw = get_robot_yaw()
cos_y = cos(robot_yaw)
sin_y = sin(robot_yaw)

path_obs = []
for point in path_window:
    # Трансляция
    dx = point[0] - robot_pos[0]
    dy = point[1] - robot_pos[1]
    
    # Ротация
    x_body = cos_y * dx + sin_y * dy
    y_body = -sin_y * dx + cos_y * dy
    
    # Clamp в [-3, 3]
    x_body = clamp(x_body, -3, 3)
    y_body = clamp(y_body, -3, 3)
    
    path_obs.extend([x_body, y_body])

# Итого: 24 элемента
```

---

### Индекс 33-34: Предыдущие действия (2 элемента)

| Индекс | Имя | Значение | Диапазон | Тип |
|--------|-----|----------|----------|-----|
| **33** | `prev_action[0]` | Линейная скорость | [-1, 1] | normalized |
| **34** | `prev_action[1]` | Угловая скорость | [-1, 1] | normalized |

**На робота эти преобразуются в реальные управления:**
```python
# Нормализованные действия → физические скорости
v_real = prev_action[0] * max_lin_vel      # max_lin_vel = 0.5 м/с
w_real = prev_action[1] * max_ang_vel      # max_ang_vel = 1.5 рад/с

# Применить к дифференциальному приводу
left_wheel_vel = (v_real - w_real * track_width / 2) / wheel_radius
right_wheel_vel = (v_real + w_real * track_width / 2) / wheel_radius
```

---

### Индекс 35-40: История компонентов награды (6 элементов)

| Индекс | Имя | Источник | Диапазон | Смысл |
|--------|-----|----------|----------|-------|
| **35** | `reward_progress` | Δs / max_distance | [-0.1, 0.1] | Прогресс по пути |
| **36** | `reward_path_error` | -d² × weight | [-0.1, 0.1] | Штраф за ошибку |
| **37** | `reward_heading` | cos(ψ) × gate | [-0.05, 0.05] | Бонус за ориентацию |
| **38** | `reward_proximity` | proximity penalty | [-0.5, 0] | Приближение к препятствиям |
| **39** | `reward_collision` | 0 или -60 | [-60, 0] | Столкновение |
| **40** | `reward_reverse` | reverse penalty | [0, 0] | (отключено) |

**На роботе эти считаются из предыдущего шага:**
```python
# Просто сохранять компоненты награды от предыдущего шага
prev_reward_components = [
    progress_reward,
    path_error_reward,
    heading_reward,
    proximity_reward,
    collision_reward,
    reverse_reward
]
```

---

## 🗺️ COSTMAP (Local Occupancy Grid)

### Размеры:
```
Форма: (3, 100, 100)
├─ Каналы (3): Три последних frames (временной контекст)
└─ Пиксели (100×100): Локальная сетка вокруг робота

Физические размеры:
├─ Размер окна: 5.0 м × 5.0 м
├─ Разрешение: 0.05 м/пиксель
├─ Диапазон вокруг робота: [-2.5, +2.5] м по X и Y
```

### Координаты пиксела:
```
costmap[c, i, j] где:
- c ∈ [0, 2]: канал (фрейм)
- i ∈ [0, 99]: строка (Y: от +2.5 до -2.5 м)
- j ∈ [0, 99]: столбец (X: от -2.5 до +2.5 м)

Center пиксела [i, j]:
x_body = (j - 49.5) * 0.05  ≈ [-2.475, +2.475] м
y_body = (49.5 - i) * 0.05  ≈ [-2.475, +2.475] м
```

### Значения в costmap (после нормализации):

| Значение | Значение до нормализации | Смысл |
|----------|-------------------------|-------|
| **0.0** | 0 | Свободное пространство |
| **0.0 - 1.0** | 1-253 | Инфлятированные препятствия (градиент) |
| **1.0** | 254 | Летальное препятствие |
| **-1.0** | 255 | Неизвестное пространство (не сканировано) |

### Как построить costmap на роботе:

```python
import numpy as np
from scipy.ndimage import distance_transform_edt

class LocalCostmapBuilder:
    def __init__(self, grid_size_m=5.0, resolution=0.05, inflation_radius=0.55):
        self.grid_size_m = grid_size_m
        self.resolution = resolution
        self.grid_width = int(grid_size_m / resolution)  # 100
        self.half_size = grid_size_m / 2.0  # 2.5 м
        self.inflation_radius_cells = int(inflation_radius / resolution)  # 11 пиксели
        
        # Буфер истории (3 фрейма)
        self.grid_history = np.full(
            (3, self.grid_width, self.grid_width),
            255,  # Инициализация как неизвестное
            dtype=np.uint8
        )
    
    def update(self, lidar_ranges, robot_yaw):
        """
        Обновить costmap на основе LiDAR данных.
        
        Args:
            lidar_ranges: массив расстояний [num_rays]
            robot_yaw: текущий угол робота в world frame
        """
        # Создать новую сетку (по умолчанию неизвестная)
        grid = np.full((self.grid_width, self.grid_width), 255, dtype=np.uint8)
        
        # LiDAR в robot frame
        lidar_angles = np.linspace(-100, 100, len(lidar_ranges)) * np.pi / 180  # примерно
        
        # Маркировать свободное пространство и препятствия
        max_range = 18.0
        for angle, range_m in zip(lidar_angles, lidar_ranges):
            if range_m > max_range:
                range_m = max_range
            
            # Вычислить координаты в robot frame
            x = range_m * np.cos(angle)
            y = range_m * np.sin(angle)
            
            # Трансформировать в grid индексы
            col = int((x + self.half_size) / self.resolution)
            row = int((y + self.half_size) / self.resolution)
            
            if 0 <= row < self.grid_width and 0 <= col < self.grid_width:
                grid[row, col] = 0  # Свободное пространство
        
        # Маркировать препятствия (в конце лучей)
        for angle, range_m in zip(lidar_angles, lidar_ranges):
            if range_m < self.half_size:  # Внутри grid
                x = range_m * np.cos(angle)
                y = range_m * np.sin(angle)
                
                col = int((x + self.half_size) / self.resolution)
                row = int((y + self.half_size) / self.resolution)
                
                if 0 <= row < self.grid_width and 0 <= col < self.grid_width:
                    grid[row, col] = 254  # Летальное препятствие
        
        # Инфляция препятствий (морфологическое расширение)
        lethal_mask = grid == 254
        if np.any(lethal_mask):
            # Простая инфляция: расстояние до ближайшего препятствия
            distance = distance_transform_edt(~lethal_mask)
            
            # Exponential decay inflation (Nav2-like)
            inflation_cost = 253 * np.exp(-10.0 * (distance * self.resolution - 0.0))
            
            inflated = np.maximum(grid, inflation_cost.astype(np.uint8))
            grid = inflated
        
        # Обновить историю (сдвиг на 1)
        self.grid_history = np.roll(self.grid_history, shift=1, axis=0)
        self.grid_history[0] = grid
        
        return self.grid_history
    
    def normalize(self):
        """Нормализовать costmap в [-1, 1]."""
        grid_obs = self.grid_history.astype(np.float32)
        
        # Замена неизвестного на -1.0
        grid_obs = np.where(
            grid_obs == 255,
            -1.0,
            grid_obs / 254.0  # Остальное в [0, 1]
        )
        
        return grid_obs
```

---

## 🔄 Pipeline на реальном роботе

### Step 1: Собрать raw данные

```python
class RobotObservationCollector:
    def collect_raw(self):
        # Получить состояние
        robot_pos_w = self.get_position()  # [x, y] в world frame
        robot_yaw = self.get_yaw()          # в world frame
        odom = self.get_odometry()          # velocities в world frame
        lidar_msg = self.get_lidar()        # LiDAR scan
        
        # Получить путь (из навигатора)
        global_path = self.nav_stack.get_plan()  # list of poses
        
        return {
            'robot_pos_w': robot_pos_w,
            'robot_yaw': robot_yaw,
            'odometry': odom,
            'lidar': lidar_msg,
            'path': global_path
        }
```

### Step 2: Построить vec и costmap

```python
def build_observation(raw_data):
    """Построить наблюдение согласно Isaac Sim формату."""
    
    # ========== VEC часть ==========
    vec = np.zeros(41, dtype=np.float32)
    
    # Indices 0-1: Direction to path
    robot_pos = raw_data['robot_pos_w']
    robot_yaw = raw_data['robot_yaw']
    global_path = raw_data['path']
    
    curr_idx = find_closest_path_index(robot_pos, global_path)
    next_point = global_path[min(curr_idx + 1, len(global_path) - 1)]
    path_direction_w = normalize([next_point.x - robot_pos[0],
                                   next_point.y - robot_pos[1]])
    
    robot_forward = [cos(robot_yaw), sin(robot_yaw)]
    vec[0] = np.dot(robot_forward, path_direction_w)  # dot_cmd
    vec[1] = np.cross(robot_forward, path_direction_w)  # cross_cmd
    
    # Indices 2-4: Velocities in robot frame
    vx_w = raw_data['odometry'].linear.x
    vy_w = raw_data['odometry'].linear.y
    wz = raw_data['odometry'].angular.z
    
    cos_y, sin_y = cos(robot_yaw), sin(robot_yaw)
    vec[2] = cos_y * vx_w + sin_y * vy_w  # vx robot
    vec[3] = -sin_y * vx_w + cos_y * vy_w  # vy robot
    vec[4] = wz  # wz
    
    # Indices 5-6: Path errors
    closest_point = global_path[curr_idx]
    error_w = np.array([robot_pos[0] - closest_point.x,
                        robot_pos[1] - closest_point.y])
    error_body = np.array([cos_y * error_w[0] + sin_y * error_w[1],
                           -sin_y * error_w[0] + cos_y * error_w[1]])
    
    vec[5] = np.clip(error_body[1], -2.5, 2.5)  # d_signed
    
    # heading error
    path_dir = normalize([global_path[min(curr_idx + 1, len(global_path)-1)].x - closest_point.x,
                          global_path[min(curr_idx + 1, len(global_path)-1)].y - closest_point.y])
    path_heading = atan2(path_dir[1], path_dir[0])
    vec[6] = normalize_angle(robot_yaw - path_heading)  # psi
    
    # Indices 7-8: Nearest LiDAR obstacle
    lidar_ranges = np.array(raw_data['lidar'].ranges[:200])  # ~200 лучей
    min_idx = np.argmin(lidar_ranges)
    min_range = np.clip(lidar_ranges[min_idx], 0, 18)
    min_angle = raw_data['lidar'].angle_min + min_idx * raw_data['lidar'].angle_increment
    
    vec[7] = np.clip(min_range * cos(min_angle), -18, 18)  # obs_x
    vec[8] = np.clip(min_range * sin(min_angle), -18, 18)  # obs_y
    
    # Indices 9-32: Path window (12 points)
    path_idx = 9
    for i in range(12):
        idx = min(curr_idx + i, len(global_path) - 1)
        point = global_path[idx]
        
        dx = point.x - robot_pos[0]
        dy = point.y - robot_pos[1]
        
        x_body = cos_y * dx + sin_y * dy
        y_body = -sin_y * dx + cos_y * dy
        
        vec[path_idx] = np.clip(x_body, -3, 3)
        vec[path_idx + 1] = np.clip(y_body, -3, 3)
        path_idx += 2
    
    # Indices 33-34: Previous actions (храните с предыдущего шага)
    vec[33] = self.last_action[0]
    vec[34] = self.last_action[1]
    
    # Indices 35-40: Previous reward components (из предыдущего шага)
    vec[35:41] = self.last_reward_components
    
    # ========== COSTMAP часть ==========
    costmap_builder = LocalCostmapBuilder()
    grid_obs_raw = costmap_builder.update(lidar_ranges, robot_yaw)
    costmap = costmap_builder.normalize()  # (3, 100, 100) в [-1, 1]
    
    return {
        "vec": vec,
        "costmap": costmap
    }
```

### Step 3: Подать в модель

```python
def flatten_observation(obs):
    """Флаттен Dict в single tensor (как делает skrl)."""
    # skrl flatten в sorted key order: costmap first, then vec
    costmap_flat = obs['costmap'].reshape(-1)  # 30,000 элементов
    vec_flat = obs['vec']  # 41 элемент
    
    state_flat = np.concatenate([costmap_flat, vec_flat])  # 30,041 элементов
    
    # Применить RunningStandardScaler нормализацию!
    # (используйте те же mean/std, что и при обучении)
    state_norm = (state_flat - self.scaler_mean) / (self.scaler_std + 1e-8)
    
    return state_norm

def run_inference(obs):
    """Пропустить через модель."""
    state_flat = flatten_observation(obs)
    
    # Стек RNN состояний (если используется GRU)
    rnn_state = self.rnn_state  # сохранять между шагами!
    
    # Forward pass через модель
    with torch.no_grad():
        action_mean, rnn_state = self.policy_net(
            torch.from_numpy(state_flat).float()[None, :],
            rnn_state
        )
    
    self.rnn_state = rnn_state  # Сохранить для следующего шага
    
    # Денормализовать действия
    action = action_mean.numpy()[0]  # [-1, 1]
    
    return action
```

---

## ⚠️ Критические моменты для деплоя

### 1. **RunningStandardScaler State**

На обучении модель видит нормализованные состояния. Вы **ДОЛЖНЫ** сохранить:
- `scaler_mean` (30,041 элементов)
- `scaler_std` (30,041 элементов)

```python
# Сохранить при обучении
scaler_state = {
    'mean': agent._state_preprocessor.running_mean.cpu().numpy(),
    'std': agent._state_preprocessor.running_std.cpu().numpy()
}
torch.save(scaler_state, 'scaler_state.pt')

# Загрузить на роботе
scaler_state = torch.load('scaler_state.pt')
self.scaler_mean = scaler_state['mean']
self.scaler_std = scaler_state['std']
```

### 2. **RNN Состояние**

Если используется GRU/LSTM (рекуррентные модели), нужно:
- Инициализировать `rnn_state = None` при начале эпизода (reset)
- **Сохранять** `rnn_state` между шагами
- Не очищать его внутри эпизода!

```python
def reset_episode(self):
    self.rnn_state = None  # Reset на начало эпизода

def step(self, obs):
    action = self.run_inference(obs)
    # rnn_state automatically updated inside run_inference
    return action
```

### 3. **Costmap Нормализация**

```python
# ✓ ПРАВИЛЬНО (как в обучении):
grid_obs = torch.where(
    grid_obs == 255,          # unknown cost
    torch.tensor(-1.0),       # -> -1.0
    grid_obs / 254.0          # rest -> [0, 1]
)

# ✗ НЕПРАВИЛЬНО:
grid_obs = grid_obs / 255.0  # неизвестное будет ≈ 1.0 (как препятствие!)
```

### 4. **Контролируемые значения**

Действия после инференции [-1, 1] трансформируются в реальные скорости:
```python
v_real = action[0] * 0.5   # max_lin_vel
w_real = action[1] * 1.5   # max_ang_vel
```

Убедитесь, что эти значения соответствуют **конфигу при обучении**!

---

## 🗂️ Config для сохранения на робот

```yaml
# deployment_config.yaml
observation_space:
  vec:
    dim: 41
    layout:
      dot_cmd: [0, 1]
      cross_cmd: [1, 2]
      velocities: [2, 5]
      path_errors: [5, 7]
      nearest_obstacle: [7, 9]
      path_window: [9, 33]  # 12 points * 2
      prev_action: [33, 35]
      prev_reward: [35, 41]  # 6 components
  
  costmap:
    shape: [3, 100, 100]
    cell_size_m: 0.05
    grid_size_m: 5.0
    normalize: true
    unknown_value: -1.0

normalization:
  type: "RunningStandardScaler"
  scaler_mean_shape: [30041]
  scaler_std_shape: [30041]
  
kinematics:
  max_lin_vel: 0.5  # m/s
  max_ang_vel: 1.5  # rad/s
  wheel_radius: 0.03  # m
  track_width: 0.242  # m
  
lidar:
  max_range: 18.0  # m
  num_rays: ~200
  fov_deg: 200  # -100 to +100
  
path:
  length_m: 6.0
  point_spacing_m: 0.10
  window_size: 12  # points ahead
```

---

## ✅ Чеклист перед деплоем

- [ ] Сохранен `scaler_mean` и `scaler_std`
- [ ] Сохранены параметры модели (`.pt` файл)
- [ ] Costmap нормализация: неизвестное → -1.0, остальное / 254
- [ ] RNN состояние инициализируется на reset
- [ ] Velocities трансформированы в robot frame
- [ ] LiDAR углы соответствуют вашему сканеру
- [ ] Путь в robot frame (координаты относительно робота)
- [ ] Действия масштабированы: v = action[0] × 0.5, w = action[1] × 1.5
- [ ] Протестирован на 1-2 эпизодах перед полным деплоем

