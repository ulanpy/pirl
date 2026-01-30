# Используем актуальный образ Isaac Sim 5.1.0
FROM nvcr.io/nvidia/isaac-sim:5.1.0

# Настройка переменных окружения
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y

USER root
# Установка системных зависимостей для Isaac Lab
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    libglib2.0-0 \
    ncurses-term \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /workspace
RUN chown -R 1234:1234 /workspace

USER 1234

# Клонируем Isaac Lab (ветка main для совместимости с 5.1.0)
RUN git clone https://github.com/isaac-sim/IsaacLab.git

# Установка Isaac Lab
WORKDIR /workspace/IsaacLab

# Создаем символическую ссылку на Isaac Sim внутри папки Isaac Lab
# Это критически важно для работы скрипта ./isaaclab.sh
RUN ln -s /isaac-sim _isaac_sim

# Используем TERM=xterm и неинтерактивный режим для установки
RUN TERM=xterm ./isaaclab.sh --install

# Устанавливаем дополнительные библиотеки
RUN ./isaaclab.sh -p -m pip install skrl wandb onnx

# Настройка путей
ENV ISAACSIM_PATH=/isaac-sim
ENV PATH="/workspace/IsaacLab:${PATH}"
