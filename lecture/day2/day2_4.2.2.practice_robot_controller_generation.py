# Initialize SimulationAPP
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
my_world = World(stage_units_in_meters=1.0)

# Initialize the Scene
scene = my_world.scene
scene.add_default_ground_plane()

# Add cube
cube = DynamicCuboid(
                prim_path = "/World/cube",
                name = "cube",
                position = [0.5, 0.5, 0.1],
                color = np.array([0, 0, 1]),
                size = 0.04,
                mass = 0.01,
            )
scene.add(cube)

# Add robot
robot_usd_path = os.path.join(lecture_dir, "utils/assets/ur5e_handeye_gripper.usd")

my_robot = UR5eHandeye(
    prim_path="/World/ur5e", # should be unique
    name="my_ur5e", # should be unique, used to access the object 
    usd_path=robot_usd_path,
    activate_camera=False,
)
scene.add(my_robot)

########################################################################

################### Pick place controller 생성 ##########################
# Add Controller
my_controller = BasicManipulationController(
    # Controller의 이름 설정
    
    # 로봇 모션 controller 설정
    
    # 로봇의 gripper 설정
    
    # phase의 진행 속도 설정
    events_dt=[0.008],
)

# robot control(PD control)을 위한 instance 선언

# Simulation Loop

########################################################################

while simulation_app.is_running():
    # 생성한 world 에서 physics simulation step​
    my_world.step(render=True)

            
# 시뮬레이션 종료
simulation_app.close()
