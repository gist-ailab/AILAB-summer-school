# SimulationApp을 사용하여 시뮬레이션을 시작하고 종료하는 코드입니다.

# SimulationApp
"""SimulationApp:: Helper class to launch Omniverse Toolkit.

Omniverse loads various plugins at runtime which cannot be imported unless
the Toolkit is already running. Thus, it is necessary to launch the Toolkit first from
your python application and then import everything else.

detail: omni.isaac.kit.simulation_app

"""
from omni.isaac.kit import SimulationApp

config = {
    'width': 640,
    'height': 480,
    'headless': False
}
simulation_app = SimulationApp(config)
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)
import carb
import omni.isaac.core.utils.carb as carb_utils
settings = carb.settings.get_settings()
carb_utils.set_carb_setting(settings, "/persistent/isaac/asset_root/default", "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1")

# After simulation is started, you can access the other functions from the isaac sim library
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid       
from omni.isaac.sensor import Camera

import time

# SimulationApp Loop, SimulationApp이 실행 중인 동안 계속 실행
time = 0
while simulation_app.is_running():
    simulation_app.update()
    print(f"Simulation Application is Running... {time}", end="\r")
    time += 1

# SimulationApp Close
simulation_app.close()
print("Simulation is Closed")

