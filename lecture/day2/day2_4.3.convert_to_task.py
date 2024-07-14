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

# Import isaacsim python API
from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.tasks import BaseTask
import omni.usd
from pxr import Gf, UsdGeom

# Import necessary libraries
import sys, os
import random
import numpy as np

# Import Custom libraries
lecture_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(lecture_dir)
from utils.robots.ur5e_handeye import UR5eHandeye
from utils.controllers.RMPFflow_pickplace import RMPFlowController

class CustomTask(BaseTask):
    """ Convert Following Target Example to Task Class

    Task is used to manage the scene and the objects in the scene.

    If you want to task based standalone application

    1. Create a task class that inherits from BaseTask
    2. Implement the methods of the task class which are used inside the World class
    - name
        - key value of the task
        - should be unique
    - set_up_scene
    - get_observations
    - calculate_metrics
    - is_done
    ...

    3. Add the task to the world and run the simulation loop

    >>> my_world = World(stage_units_in_meters=1.0)
    >>> my_task = CustomTask(name="my_task")
    >>> my_world.add_task(my_task)
    >>> my_world.reset()
    >>> while simulation_app.is_running():
    >>>     my_world.step(render=True)
    >>>     if my_world.is_playing():
    >>>         observations = my_world.get_observations()
    >>>         # some logic using the observations
    >>>         if my_task.is_done(observations):
    >>>             break
    >>> simulation_app.close()
    
    
    """

    def __init__(self, name):
        BaseTask.__init__(self, name=name)

        self._robot = None
        self._target_cube = None
        self._rmpflow_controller = None
        self._robot_controller = None

        self._robot_usd_path = os.path.join(lecture_dir, "utils/assets/ur5e_handeye_gripper.usd")
        
        return
    
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        # Add cube
        self._target_cube = VisualCuboid(
                        prim_path = "/World/cube",
                        name = "cube",
                        position = [0.5, 0.5, 0.1],
                        color = np.array([0, 0, 1]),
                        size = 0.04,
                    )
        scene.add(self._target_cube)

        # Add robot
        self._robot = UR5eHandeye(
            prim_path="/World/ur5e", # should be unique
            name="my_ur5e", # should be unique, used to access the object 
            usd_path=self._robot_usd_path,
            activate_camera=False,
        )        
        scene.add(self._robot)

        self._rmpflow_controller = RMPFlowController(
            name="end_effector_controller_cspace_controller", robot_articulation=self._robot, attach_gripper=True
        )
        self._robot_controller = self._robot.get_articulation_controller()

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
        self.wrist = UsdGeom.Xformable(wrist)
        self.left_finger = UsdGeom.Xformable(left_finger)
        self.right_finger = UsdGeom.Xformable(right_finger)


    def get_controller(self):
        return self._rmpflow_controller
    
    def get_robot_controller(self):
        return self._robot_controller

    def get_observations(self) -> dict:        
        target_position, target_orientation = self._target_cube.get_world_pose()
        
        # Get the local transformation matrix
        wrist_transform = self.wrist.ComputeLocalToWorldTransform(0)  
        left_finger_transform = self.left_finger.ComputeLocalToWorldTransform(0)
        right_finger_transform = self.right_finger.ComputeLocalToWorldTransform(0)
              
        # Extract the translation (position) from the transformation matrix
        wrist_position = wrist_transform.ExtractTranslation()
        left_finger_position = left_finger_transform.ExtractTranslation()
        right_finger_position = right_finger_transform.ExtractTranslation()

        offset = wrist_position - (left_finger_position+right_finger_position)/2
        target_position +=  offset # For Gipper offset
    
        return {
            "target_position": target_position
        }
    

    def is_done(self) -> bool:
        return False # Not Done while Application is running



# World 선언
my_world = World(stage_units_in_meters=1.0)

# task 선언
my_task = CustomTask(name="my_task")
my_world.add_task(my_task)

# Reset the world = Scene Setup
my_world.reset() 

# Get Controller
my_controller = my_task.get_controller()
articulation_controller = my_task.get_robot_controller()

while simulation_app.is_running():
    # 생성한 world 에서 physics simulation step
    my_world.step(render=True)
    
    # world가 동작하는 동안 작업 수행
    if my_world.is_playing():
        observations = my_world.get_observations()
        
        # some logic using the observations
        target_position = observations["target_position"]

        # 선언한 my_controller를 사용하여 action 수행
        actions = my_controller.forward(
            target_end_effector_position=target_position,
        )
        articulation_controller.apply_action(actions)
        
        # task의 끝남 여부를 확인
        if my_task.is_done():
            break

simulation_app.close()