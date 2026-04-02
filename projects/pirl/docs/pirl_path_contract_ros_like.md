# PIRL Path Contract (path-geometry reward, no replanning)

Этот документ фиксирует контракт между генератором пути и RL-контроллером в среде `pirl` после удаления look-ahead логики.

## 1) Роли в среде

- **Планнер (упрощенно)**: один раз генерирует глобальный путь в world frame при `reset()`.
- **Контроллер (каждый тик)**: берет ближайшую точку пути, строит локальное окно пути в frame робота и выдает действие.
- **Replanning отсутствует** в рамках эпизода: путь фиксирован до следующего `reset()`.

Реализация:
- путь: `source/pirl/pirl/tasks/direct/pirl/pirl_env_path.py`
- observation/reward: `source/pirl/pirl/tasks/direct/pirl/pirl_env.py`
- параметры: `source/pirl/pirl/tasks/direct/pirl/pirl_env_cfg.py`

## 2) Представление пути

Для каждого env:
- `path_points_w`: `(num_envs, path_num_points, 2)` — точки ломаной в world XY.
- `path_s`: `(num_envs, path_num_points)` — накопленная длина дуги вдоль ломаной.

Расчет `path_s`:
- `path_s[0] = 0`
- `path_s[i] = path_s[i-1] + ||p_i - p_{i-1}||`

Это и есть координата прогресса `s` вдоль заданной траектории.

## 3) Опорная точка пути (`curr_idx`)

На каждом тике:
1. Ищется ближайшая к роботу точка пути среди индексов `>= path_idx` (monotonic prune).
2. `path_idx = max(path_idx, nearest_idx)` — индекс не откатывается назад.

Смысл:
- `curr_idx` — единая опора для reward и локального path-window.

## 4) Локальный state пути (sliding window)

В observation передается не весь глобальный путь, а локальный фрагмент:
- берется сегмент `[curr_idx, curr_idx+1, ..., curr_idx+path_segment_len-1]`,
- индексы clamp к последней точке,
- сегмент переводится в систему координат робота и флеттится в `vec`.

Это и есть скользящее окно: по мере роста `curr_idx` окно автоматически сдвигается вперед.

## 5) Reward

Используется только геометрия пути:

`r = w1 * (s_t - s_{t-1}) - w2 * d_path + w3 * cos(Δθ)`

Где:
- `s_t` — `path_s[curr_idx]` на текущем тике,
- `d_path` — расстояние от робота до ближайшей точки пути (по текущему `curr_idx`),
- `Δθ = θ_robot - θ_path`,
- `θ_path` — угол касательной к пути около `curr_idx` (по соседним точкам).

Важно:
- reward больше не зависит от внешней "движущейся цели";
- убрана проблема "убегающей морковки";
- метрика прогресса измеряется вдоль фиксированной траектории.

## 6) Что подается в state

Текущий `vec` включает:
- кинематику робота (`dot`, `cross`, `v_x`, `v_y`, `w_z`),
- локальные относительные координаты точек пути (sliding window),
- `prev_action`,
- `prev_reward_components = [progress, path_error, heading]`.

## 7) Ключевые параметры

- `path_length_m`
- `path_point_spacing_m`
- `path_num_points = round(path_length_m / path_point_spacing_m) + 1`
- `path_segment_len` (размер sliding window в точках)
- `rew_scale_progress` (`w1`)
- `rew_scale_path_error` (`w2`)
- `rew_scale_heading` (`w3`)

## 8) Практический смысл

- Reward оптимизирует именно движение по траектории, а не погоню за эвристической точкой.
- State дает локальную форму пути впереди робота.
- Поведение становится стабильнее для RL и лучше согласовано с path-following постановкой.
