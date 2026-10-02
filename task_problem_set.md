# PIRL Task Problem Set

Рабочий документ для формализации task family и curriculum. Это не список
случайных сцен: каждый раздел описывает отдельную проверяемую способность
локального контроллера. Конкретные distributions, reward weights и критерии
сложности будут уточняться итеративно.

## Общий контракт task family

Во всех задачах сохраняются неизменными:

- один дифференциальный TurtleBot3 Burger;
- действие: нормализованные `[linear_velocity, yaw_velocity]`;
- наблюдение: локальная LiDAR hit-map, состояние робота и окно reference path;
- цель: безопасно и эффективно завершить заданный reference path;
- policy — локальный контроллер, а не глобальный planner или SLAM-система.

Сценарий `z` выбирается при reset. Curriculum меняет распределение сценариев,
но не observation/action contract.

## 1. Basic motion control

### Capability

Выполнять выполнимые линейные и угловые команды устойчиво: ехать, поворачивать,
останавливаться и восстанавливаться после начальной ошибки позы.

### Scenario variables

- начальная поза и heading error;
- требуемое направление и длина короткого отрезка;
- допустимая скорость и кривизна движения;
- позднее: задержка привода и простые возмущения.

### Success / failure

- достигнут короткий goal или завершён простой path segment;
- ограничены cross-track error, heading error и oscillation;
- failure: timeout, выход за допустимую область, collision.

### Open questions

- Нужна ли эта ветка как отдельный training stage или только как
  контролируемый smoke/evaluation benchmark?
- Как измерять smoothness: variation действий, jerk команд или траектория колёс?

## 2. Reference path following

### Capability

Продвигаться вдоль непрерывной polyline reference path, сохраняя положение в
допустимом corridor вокруг неё. Path points — представление будущей траектории,
а не единственные дискретные цели.

### Scenario variables

- длина пути, кривизна, S-повороты и частота смены направления;
- начальное lateral/heading отклонение;
- ширина допустимого path corridor;
- spacing и horizon path points в observation.

### Success / failure

- path completion и положительный progress по arc length `s`;
- cross-track и heading error в заданных пределах;
- failure: timeout, слишком большое отклонение от path, collision.

### Open questions

- Какие значения corridor соответствуют реалистичному запасу вокруг Burger?
- Когда deviation от path — допустимый обход, а когда failure?

## 3. Static obstacle avoidance

### Capability

Безопасно обходить неподвижную геометрию, временно отклоняться от reference
path при необходимости и затем возвращаться к нему (`rejoin`).

### Scenario variables

- форма: цилиндр, box, длинная стена, группа примитивов;
- положение и ориентация относительно path;
- overlap с path и минимальный clearance;
- число доступных сторон обхода;
- узкий проход, S-обход, объект после поворота, частичная окклюзия;
- топология: один проход, два локально различимых варианта, тупик.

### Success / failure

- completion без collision с контролируемым минимальным clearance;
- после обхода восстановлен progress на reference path;
- failure: collision, timeout, уход в тупик/за пределы сценария.

### Invariants for the generator

- Сначала задаётся навигационная ситуация, затем строится её геометрия.
- Обход должен существовать, быть видимым в доступном local horizon и быть
  кинематически выполнимым для Burger.
- Статичная сцена либо уже учтена global path planner, либо намеренно
  перекрывает reference path для обучения local rejoin — эти режимы не
  смешиваются неявно.

### Open questions

- Какие топологии честно решаемы только локальной policy без global map?
- Нужен ли отдельный failure criterion для insufficient clearance?

## 4. Dynamic obstacle avoidance and interaction

### Capability

Реагировать на видимые движущиеся препятствия, не сталкиваться, не застывать
без причины и продолжать движение после освобождения прохода. На продвинутом
уровне использовать последовательность наблюдений для anticipation, а не
реагировать лишь на текущую позицию объекта.

### Scenario variables

- число объектов, размер, скорость, ускорение и stop/go поведение;
- направление: попутное, встречное, пересекающее path, боковое;
- time-to-collision и начальная дистанция;
- траектория: детерминированная, stochastic, смена намерения;
- плотность, взаимные окклюзии и порядок появления в LiDAR range.

### Curriculum levels

1. Один объект с постоянной скоростью, заранее видимый.
2. Один пересекающий объект: policy должна уступить или объехать.
3. Stop/go и изменение скорости: различить временно занятый и свободный проход.
4. Несколько объектов и неоднозначные, но физически наблюдаемые траектории.

### Success / failure

- collision rate, path completion, time-to-completion;
- minimum clearance и время неоправданной остановки;
- отдельная метрика frozen robot: безопасен, но не делает progress.

### Open questions

- Какие motion models препятствий считаем реалистичными для первого этапа?
- Как отделить anticipation от простой реактивной политики в evaluation?

## 5. Navigation under partial observability and uncertainty

### Capability

Принимать безопасные решения при неполном или шумном LiDAR-наблюдении:
снижать скорость и не делать необоснованно рискованный манёвр, когда будущее
пространство не наблюдаемо. Память policy должна помогать сохранять временной
контекст, но не требовать невозможного точного предсказания скрытого объекта.

### Scenario variables

- ограниченный range/FoV, сенсорный шум, dropout и задержка;
- поворот с невидимой областью за углом;
- временная окклюзия dynamic obstacle другим объектом;
- степень наблюдаемости: полностью видимый, частично видимый, исчезнувший.

### Success / failure

- safety при позднем появлении препятствия;
- разумное снижение скорости вместо collision;
- отсутствие постоянного freeze в безопасной пустой сцене;
- generalization на unseen noise/occlusion parameters.

### Open questions

- Какие uncertainty cues физически доступны реальному Burger и ROS pipeline?
- Нужна ли отдельная observation feature о качестве/давности сенсорных данных?

## 6. Robustness and sim-to-real transfer

### Capability

Сохранять тот же навигационный навык при умеренном расхождении dynamics,
сенсоров и исполнения между Isaac Sim и физическим TurtleBot. Это ось
robustness, а не отдельная цель навигации.

### Scenario variables

- wheel radius/base, mass, friction и damping;
- actuator delay, action latency и command-rate variation;
- wheel slip/skid и внешние возмущения;
- LiDAR noise, range bias, missed hits и pose/path error;
- domain randomization distribution и held-out OOD distribution.

### Success / failure

- сравнение success/collision/progress между nominal и randomized domains;
- degradation curve по уровню perturbation;
- evaluation на hold-out combinations, которых не было в train distribution.

### Open questions

- Какие параметры можно калибровать по реальному Burger до training?
- Какие randomizations нужны сейчас, а какие преждевременны до устойчивого
  nominal policy?

## Explicitly out of current scope

- Global SLAM, map ageing и data association: это отдельный слой системы, а не
  задача текущего local policy.
- Social navigation: станет отдельной веткой только после появления людей как
  семантических агентов, формальных social norms и измеримых метрик.
- Multi-robot coordination: потребует другого observation/action и, вероятно,
  multi-agent formulation.
