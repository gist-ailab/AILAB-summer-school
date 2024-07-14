# Initialize SimulationAPPs
from omni.isaac.kit import SimulationApp
config = {
    'width': 640,
    'height': 480,
    'headless': False
}
simulation_app = SimulationApp(config)
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)
# (optional) Set default nucleus path for docker user
import carb
import omni.isaac.core.utils.carb as carb_utils
settings = carb.settings.get_settings()
carb_utils.set_carb_setting(settings, "/persistent/isaac/asset_root/default", "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1")

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

import sys, os
import numpy as np
lecture_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(lecture_dir)

from utils.robots.ur5e_handeye import UR5eHandeye
from utils.controllers.RMPFflow_pickplace import RMPFlowController
from utils.controllers.basic_manipulation_controller import BasicManipulationController

############### 로봇의 기본적인 매니퓰레이션 동작을 위한 환경 설정 ################

# World 선언

# Initialize the Scene

# Add cube

# Add robot

########################################################################

while simulation_app.is_running():
    # 생성한 world 에서 physics simulation step​
    my_world.step(render=True)

            
# 시뮬레이션 종료
simulation_app.close()
