import os
import argparse
from isaacsim import SimulationApp

# В этом режиме мы не передаем headless в SimulationApp, 
# так как запуск через runheadless.sh сам всё настроит
simulation_app = SimulationApp({
    "window_width": 1280,
    "window_height": 720,
    "headless": True,
    "display_options": 0, # Отключает создание окон на уровне драйвера
})

# Принудительно включаем расширение стриминга
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.kit.livestream.webrtc")

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.world import World

def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    extension_dir = os.path.dirname(current_dir)
    usd_path = os.path.join(extension_dir, "assets", "usd", "jettank.usd")

    if not os.path.exists(usd_path):
        print(f"[ERROR] USD file not found: {usd_path}")
        simulation_app.close()
        return

    print(f"[INFO] Spawning robot from: {usd_path}")
    prim_utils.create_prim(
        prim_path="/World/Robot",
        prim_type="Xform",
        position=[0, 0, 0.2],
        usd_path=usd_path
    )

    set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])

    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()