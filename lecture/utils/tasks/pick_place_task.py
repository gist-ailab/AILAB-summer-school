# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.tasks as tasks
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import create_prim, get_prim_path, define_prim
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.semantics import add_update_semantics
from utils.robots.ur5e_handeye import UR5eHandeye
import os, random
import numpy as np
from typing import Optional
from pxr import Gf
import omni.usd


class UR5ePickPlace(tasks.PickPlace):
    """[summary]

        Args:
            name (str, optional): [description]. Defaults to "ur5_pick_place".
            cube_initial_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_initial_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str = "ur5e_pick_place",
        objects_list: Optional[list] = None,    # import mesh file such as stl, obj, etc.
        objects_position: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = np.array([0, 0, 0.15]),
    ) -> None:
        tasks.PickPlace.__init__(self, name=name, )
        if (objects_position is None) and (objects_list is not None):
            for i in range(len(objects_list)):
                if i == 0:
                    pos_x = random.uniform(0.4, 0.7)
                    pos_y = random.uniform(0.4, 0.7)
                    pos_z = 0.1
                    self.objects_position = np.array([[pos_x, pos_y, pos_z]])
                elif i == 1:
                    pos_x = random.uniform(-0.4, -0.7)
                    pos_y = random.uniform(0.4, 0.7)
                    pos_z = 0.1
                    self.objects_position = np.concatenate((self.objects_position, np.array([[pos_x, pos_y, pos_z]])),
                                                          axis=0)
                elif i == 2:
                    pos_x = random.uniform(-0.4, -0.7)
                    pos_y = random.uniform(-0.4, -0.7)
                    pos_z = 0.1
                    self.objects_position = np.concatenate((self.objects_position, np.array([[pos_x, pos_y, pos_z]])),
                                                          axis=0)
        else:
            self.objects_position = objects_position

        self.imported_objects = objects_list
        self.imported_objects_prim_path = "/World/object"
        
        self.objects_position_list = []
        self.objects_orientation_list = []
        self.objects_name_list = []

        if self.imported_objects is None:
            cube_prim_path = "/World/Cube"
            cube_name = "cube"
            if objects_position is None:
                pos_x = random.uniform(0.3, 0.6)
                pos_y = random.uniform(0.3, 0.6)
                pos_z = 0.1
                self.objects_position = np.array([[pos_x, pos_y, pos_z]])
                self.objects_position[0][2] = self.objects_position[0][2] + 0.01

            self._object = DynamicCuboid(
                prim_path = cube_prim_path,
                name = cube_name,
                position = self.objects_position[0],
                color = np.array([0, 0, 1]),
                size = 0.04,
                mass = 0.01,
            )
            self._objects = self._object
        else:
            # size_scale = 0.2
            self._objects = objects_list

        self._offset = offset
        if target_position is None:
            target_position = np.array([0.4, -0.33, 0])
            target_position[2] = 0.05 # considering the length of the gripper tip
        self.target_position = target_position + self._offset
        return


    def set_up_scene(self, scene: Scene) -> None:
        """[summary]
        YCB objects are added to the scene. If the ycb objects are not found in the scene, 
        only the cuboid added to the scene.

        Args:
            scene (Scene): [description]
        """
        self._scene = scene
        scene.add_default_ground_plane()

        if self.imported_objects is None:   # add only cube
            self._task_object = scene.add(self._objects)
        else:
            for i in range(len(self.imported_objects)):
                self.set_usd_objects(object_number = i,
                                     object_position = self.objects_position[i],)

        self._robot = self.set_robot()
        scene.add(self._robot)
        self.set_camera()

        self._move_task_objects_to_their_frame()
        return


    def set_robot(self) -> UR5eHandeye:
        """[summary]

        Returns:
            UR5e: [description]
        """
        working_dir = os.path.dirname(os.path.realpath(__file__))   # same directory with this code
        # ur5e_usd_path = os.path.join(working_dir, "ur5e_handeye_gripper.usd")
        ur5e_usd_path = os.path.join(working_dir, "ur5e_handeye_gripper.usd")
        if os.path.isfile(ur5e_usd_path):
            pass
        else:
            raise Exception(f"{ur5e_usd_path} not found")
        
        ur5e_prim_path = find_unique_string_name(
            initial_name="/World/ur5e", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        ur5e_robot_name = find_unique_string_name(
            initial_name="my_ur5e", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        return UR5eHandeye(prim_path = ur5e_prim_path,
                           name = ur5e_robot_name,
                           usd_path = ur5e_usd_path)


    def set_usd_objects(self, object_number: int, object_position: np.ndarray) -> None:
        # https://forums.developer.nvidia.com/t/set-mass-and-physicalmaterial-properties-to-prim/229727/4
        define_prim(self.imported_objects_prim_path + f"_{object_number}")
        define_prim(self.imported_objects_prim_path + f"_{object_number}" + "/model_normalized")
        
        usd_path = self._objects[object_number]
        category = usd_path.split('/')[-3]
        if category in ['03261776', '02876657']:
            orientation = np.array([1, 0, 0, 0])
        elif category in ['02958343']:
            orientation = np.array([0.638613, 0.6882868, 0, 0.3441434])
        else:
            orientation = np.array([0.7071, 0.7071, 0, 0])
            
        
        self._task_object = add_reference_to_stage(usd_path = usd_path,
                                                   prim_path = self.imported_objects_prim_path + f"_{object_number}")
        
        rigid_prim = RigidPrim(prim_path = self.imported_objects_prim_path + f"_{object_number}",
                               position = object_position,
                               orientation=orientation,
                               name = "rigid_prim" + f"_{object_number}",
                               scale = np.array([0.4] * 3),
                               mass = 0.01)
        rigid_prim.enable_rigid_body_physics()
        
        self._scene.add(rigid_prim)

        geometry_prim = GeometryPrim(prim_path = self.imported_objects_prim_path + f"_{object_number}" + "/model_normalized",
                                     name = f"geometry_prim_{object_number}",
                                     position = object_position,
                                     orientation=orientation,
                                     scale = np.array([0.4] * 3),
                                     collision = True,
                                    )
        geometry_prim.set_collision_enabled(True)
        geometry_prim.apply_physics_material(
            PhysicsMaterial(
                prim_path = self.imported_objects_prim_path + f"_{object_number}" + f"/physics_material_{object_number}",
                static_friction = 50,
                dynamic_friction = 50,
                restitution = 0.01
            )        
        )
        
        self._scene.add(geometry_prim)

        model_prim = self._scene.get_object(name=f"geometry_prim_{object_number}")
        model_prim.set_collision_approximation('convexDecomposition')

    
    def get_params(self) -> dict:
        params_representation = dict()
        if self.imported_objects is None:
            self.position, self.orientation = self._task_object.get_local_pose()
            self.task_object_name = self._task_object.name
            params_representation[f"task_object_position_0"] = {"value": self.position, "modifiable": True}
            params_representation[f"task_object_orientation_0"] = {"value": self.orientation, "modifiable": True}
            params_representation[f"task_object_name_0"] = {"value": self.task_object_name, "modifiable": False}

        else:
            for i in range(len(self.imported_objects)):
                stage = omni.usd.get_context().get_stage()
                prim = stage.GetPrimAtPath(self.imported_objects_prim_path + f"_{i}")
                matrix = omni.usd.get_world_transform_matrix(prim)
                translate = matrix.ExtractTranslation()
                rotation = matrix.ExtractRotationQuat()
                self.position = np.array([translate[0], translate[1], translate[2]+0.03],
                                         dtype=np.float32)
                self.objects_position_list.append(self.position)
                self.orientation = np.array([rotation.imaginary[0],
                                             rotation.imaginary[1],
                                             rotation.imaginary[2],
                                             rotation.real],
                                            dtype=np.float32)
                self.objects_orientation_list.append(self.orientation)
                self.task_object_name = prim.GetName()
                self.objects_name_list.append(self.task_object_name)

                params_representation[f"task_object_position_{i}"] = {"value": self.position, "modifiable": True}
                params_representation[f"task_object_orientation_{i}"] = {"value": self.orientation, "modifiable": True}
                params_representation[f"task_object_name_{i}"] = {"value": self.task_object_name, "modifiable": False}
        
        params_representation["target_position"] = {"value": self.target_position, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation


    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()

        observation_dict = dict()
        if self.imported_objects is None:
            observation_dict = {
                                self.task_object_name: {"position": self.position,
                                                        "orientation": self.orientation,
                                                        "target_position": self.target_position,
                                                        },
                                self._robot.name: {"joint_positions": joints_state.positions,
                                                   "end_effector_position": end_effector_position,
                                                   },
                                }
        else:
            for i in range(len(self.imported_objects)):
                observation_dict[self.objects_name_list[i]] = {"position": self.objects_position_list[i],
                                                          "orientation": self.objects_orientation_list[i],
                                                          "target_position": self.target_position,
                                                          }
            observation_dict[self._robot.name] = {"joint_positions": joints_state.positions,
                                                  "end_effector_position": end_effector_position,
                                                  }
        return observation_dict
    
    
        
    def set_camera(self):
        self.camera = Camera(
            prim_path="/World/ur5e/realsense/Depth",
            frequency=10,
            resolution=(1920, 1080),
        )
        
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame() # for get depth image
        
    def get_camera(self):
        return self.camera