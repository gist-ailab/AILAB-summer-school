""" UR5e 로봇이 target을 따라가는 예제 

- 사용하는 UR5e 로봇은 연구실에서 조립한 USD 파일을 사용하며, 
- Robot Class 를 상속받아 구현한 UR5eHandeye 클래스를 사용합니다.

"""
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
from omni.isaac.core.objects import VisualCuboid
import omni.usd
from pxr import UsdGeom

import sys, os
import numpy as np

# Import Custom Class
lecture_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(lecture_dir)
from utils.robots.ur5e_handeye import UR5eHandeye
from utils.controllers.RMPFflow_pickplace import RMPFlowController

# World 선언
my_world = World(stage_units_in_meters=1.0)

# Initialize the Scene
scene = my_world.scene
scene.add_default_ground_plane()

# Add cube
cube = VisualCuboid(
                prim_path = "/World/cube",
                name = "cube",
                position = [0.5, 0.5, 0.1],
                color = np.array([0, 0, 1]),
                size = 0.04,
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

stage = omni.usd.get_context().get_stage()
# Define the prim path
wrist_path = "/World/ur5e/robotiq_arg2f_base_link"
left_finger_path = "/World/ur5e/left_inner_finger_pad"
right_finger_path = "/World/ur5e/right_inner_finger_pad"

# Get the prim using the path
wrist = stage.GetPrimAtPath(wrist_path)
left_finger = stage.GetPrimAtPath(left_finger_path)
right_finger = stage.GetPrimAtPath(right_finger_path)

# Get the Xformable API to access transformation attributes
wrist = UsdGeom.Xformable(wrist)
left_finger = UsdGeom.Xformable(left_finger)
right_finger = UsdGeom.Xformable(right_finger)

# Add Controller
my_controller = RMPFlowController(
        name="end_effector_controller_cspace_controller", robot_articulation=my_robot, attach_gripper=True
    )
# robot control(PD control)을 위한 instance 선언
articulation_controller = my_robot.get_articulation_controller()

# Simulation Loop
my_world.reset()
my_controller.reset()
# target position

while simulation_app.is_running():
    # 생성한 world 에서 physics simulation step
    my_world.step(render=True)
    ee_target_position, ee_target_orientation = cube.get_world_pose()
    
    # Get transformation
    wrist_transform = wrist.ComputeLocalToWorldTransform(0)
    left_finger_transform = left_finger.ComputeLocalToWorldTransform(0)
    right_finger_transform = right_finger.ComputeLocalToWorldTransform(0)
            
    # Extract the translation (position) from the transformation matrix
    wrist_position = wrist_transform.ExtractTranslation()
    left_finger_position = left_finger_transform.ExtractTranslation()
    right_finger_position = right_finger_transform.ExtractTranslation()

    offset = wrist_position - (left_finger_position+right_finger_position)/2
    ee_target_position += offset # For Gipper offset
    
    if my_world.is_playing():
        actions = my_controller.forward(
            target_end_effector_position=ee_target_position,
        )
        
        # 컨트롤러 내부에서 계산된 타겟 joint position값을
        # articulation controller에 전달하여 action 수행
        articulation_controller.apply_action(actions)

            
# 시뮬레이션 종료
simulation_app.close()
