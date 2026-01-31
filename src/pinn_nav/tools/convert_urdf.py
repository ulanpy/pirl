import os
from omni.isaac.kit import SimulationApp

# Запуск мини-приложения Isaac Sim для конвертации
simulation_app = SimulationApp({"headless": True})

import omni.kit.commands

def convert_urdf_to_usd(urdf_path, usd_path):
    print(f"[INFO] Converting {urdf_path} to {usd_path}...")
    
    if not os.path.exists(urdf_path):
        print(f"[ERROR] URDF file not found: {urdf_path}")
        return

    # Настройки импорта
    # Мы используем стандартные настройки, которые хорошо работают для мобильных роботов
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.make_default_prim = True
    import_config.create_physics_scene = True
    
    # Сама конвертация
    success = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        dest_path=usd_path
    )
    
    if success:
        print(f"[SUCCESS] Conversion complete! USD saved at: {usd_path}")
    else:
        print("[ERROR] Conversion failed!")

if __name__ == "__main__":
    # АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ПУТЕЙ
    # Находим папку assets относительно этого скрипта
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Поднимаемся на уровень выше до pinn_nav (если структура src/pinn_nav/tools/convert_urdf.py)
    extension_dir = os.path.dirname(current_dir)
    
    urdf_file = os.path.join(extension_dir, "assets", "urdf", "jettank.urdf")
    usd_file = os.path.join(extension_dir, "assets", "usd", "jettank.usd")
    
    print(f"[DEBUG] Current script directory: {current_dir}")
    print(f"[DEBUG] Looking for URDF at: {urdf_file}")
    
    # Пытаемся создать директорию для USD
    try:
        os.makedirs(os.path.dirname(usd_file), exist_ok=True)
    except PermissionError:
        print(f"[ERROR] Permission denied when creating: {os.path.dirname(usd_file)}")
        print("[TIP] Run 'sudo chmod -R 777 src' on your host machine to fix this.")
        simulation_app.close()
        exit(1)

    convert_urdf_to_usd(urdf_file, usd_file)
    
    simulation_app.close()
