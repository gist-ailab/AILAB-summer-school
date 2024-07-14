# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

""" Method definition for Pick and Place scene

    This task class will be used to manage the scene, the objects and the robot in the scene.

    1. Get the object mesh file path and the object position list from main python file.

    2. Implement the methods of the task class.
        - set_usd_background: add background object(basket,table) to the scene
        - set_usd_objects: add object(for grasping) to the scene
        - set_up_scene : call set_usd_background, set_usd_objects, set_robot, set_camera for setting up the scene
        - get_params : get the object position, orientation, name, robot name
        - get_observations : get the object position, orientation, robot joint position, end effector position
        - set_camera : add camera to the scene
        - get_camera : get camera
"""
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.tasks as tasks
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from omni.isaac.sensor import Camera
from utils.robots.ur5e_handeye import UR5eHandeye
import os
import numpy as np
from typing import Optional
from pxr import Gf
import omni.usd


import omni
from pxr import Usd, UsdGeom, Gf, UsdPhysics

# 3D bbox를 활용한 mesh size 측정(x,y,z)
# PRACTICE 24: get_mesh_size 메소드 작성
def get_mesh_size(prim_path):
    # 현재 stage 취득

    # prim path로 prim 취득

    # prim size를 측정하기 위한 bbox cache 생성


    # prim의 bbox 취득


    # prim size 취득

    return prim_size

class UR5ePickPlace(tasks.PickPlace):
    """
        Args:    
            name (str, optional): [description]. Defaults to "ur5_pick_place"
            objects_list: Optional[list],                : object mesh file(usd) list
            objects_position: Optional[np.ndarray],      : object position list
            background_objects_list = None,              : background object mesh file(usd) list
            background_target_position = None,           : background object position list
            offset: Optional[np.ndarray] = np.array([0, 0, 0.01]),      : mesh collison offset
            robot_position = np.array([0.0, 0.0, 1.0]),                 : robot position
    """

    def __init__(
        self,
        name: str = "ur5e_pick_place",
        objects_list: Optional[list] = None,
        objects_position: Optional[np.ndarray] = None,
        background_objects_list = None,
        background_target_position = None,
        offset: Optional[np.ndarray] = np.array([0, 0, 0.01]),
        robot_position = np.array([0.0, 0.0, 1.0]),
    ) -> None:
        tasks.PickPlace.__init__(self, name=name, )
        
        # shapenet 물체 정보
        self.objects_position = objects_position
        self.imported_objects = objects_list
        self.imported_objects_prim_path = "/World/object"
        
        self.objects_position_list = []
        self.objects_orientation_list = []
        self.objects_name_list = []

        self.background_objects_list = background_objects_list
        self.background_target_position = background_target_position
        # shapenet dataset의 좌표계를 Isaac sim 의 좌표계와 일치하도록 xzy축을 xyz로 변환하기위한 quaternion 값
        # PRACTICE 4: shapenet의 좌표계(xzy)를 isaac sim simulator의 좌표계(xyz)로 변경하기 위한 quaternion 값 채우기
        self.xzy2xyz = 

        self.robot_position = robot_position

        self._objects = objects_list

        self._offset = offset
        return

    def set_usd_background(self, object_number: int, object_position: np.ndarray) -> None:
        # initialize usd_path, obj_name, prim_path
        usd_path = self.background_objects_list[object_number]
        obj_name = usd_path.split('/')[-2]
        # PRACTICE 5: prim_path 정의하기
        background_object_prim_path =  # 1st, 2nd object
        # define prim for background object
        define_prim()   # PRACTICE 5
        define_prim()   # PRACTICE 5

        # add usd object to the scene
        # PRACTICE 6: add_reference_to_stage 함수 채우기
        add_reference_to_stage(usd_path = ,
                                prim_path = )
        
        # object scaling
        if obj_name == "table":
            scale = np.array([1.5, 1.0, 2.0])
        elif obj_name == "basket":
            scale = np.array([0.5, 0.2, 0.5])

        # temporary prim for size check
        # PRACTICE 7: size 측정을 위한 임시 RigidPrim 생성
        RigidPrim(prim_path = ,
                                position = [0,0,0], 
                                orientation=self.xzy2xyz,
                                scale = ,
                                name = ,
                                mass = 10.0,
                                density = 10.0)
        
        # get mesh size
        # PRACTICE 8: get_mesh_size 메소드 채우기
        mesh_size = get_mesh_size()
        
        # convert scale factor after mesh size normalization(1*1*1)
        if obj_name == "table":
            # 가로,세로, 높이 모두 1로 만들기
            scale[0] = scale[0] / mesh_size[0]     # x-axis scale(sim 기준)
            scale[1] = scale[1] / mesh_size[2]     # z-axis scale(sim 기준) - 높이축
            scale[2] = scale[2] / mesh_size[1]     # y-axis scale(sim 기준)
            # 테아블 높이의 중간으로 위치 조정
            object_position[2] = object_position[2] + mesh_size[2] * scale[1] / 2   # 책상높이/2
            # 고정할 위치를 물체 위치로 설정
            # PRACTICE 9: fixed_joint_postion 지정하기
            fixed_joint_postion = Gf.Vec3f()

        elif obj_name == "basket":
            # 가로,세로, 높이 모두 1로 만들기
            scale[0] = scale[0] / mesh_size[0]     # x-axis scale(sim 기준)
            scale[1] = scale[1] / mesh_size[2]    # z-axis scale(sim 기준) - 높이축
            scale[2] = scale[2] / mesh_size[1]     # y-axis scale(sim 기준)
            # 바구니 높이의 중간으로 위치 조정 + 테이블 높이 + offset
            object_position[2] = object_position[2] + mesh_size[2] * scale[1] / 2 + 1.01 #책상높이+offset
            # 고정할 위치를 물체 위치로 설정
            fixed_joint_postion = Gf.Vec3f()    # PRACTICE 9


        # rigid body to the scene: mass, velocity, force, etc.
        # PRACTICE 10: prim path, name 정의하기, scale 지정.
        rigid_prim = RigidPrim(prim_path = ,
                                position = object_position+self._offset,
                                orientation=self.xzy2xyz,
                                scale = ,
                                name = ,
                                mass = 10.0,
                                density = 10.0)      
        
        # add to the scene
        self._scene.add(rigid_prim)
        rigid_prim.enable_rigid_body_physics()
        
        
        # add geometry prim to the scene: visualizing the object, collision enabled
        # PRACTICE 11: prim path, name 지정하기. scale 지정.
        geometry_prim = GeometryPrim(prim_path = ,
                                        name = ,
                                        position = object_position+self._offset, 
                                        orientation=self.xzy2xyz,
                                        scale = ,
                                        collision = True,
                                    )
        # set physics material: surface properties
        # PRACTICE 12: prim path 지정하기.
        geometry_prim.apply_physics_material(
            PhysicsMaterial(
                prim_path = ,
                static_friction = 1.0,       # 정적마찰계수
                dynamic_friction = 1.0,      # 동적마찰계수
                restitution = 0.01          # 반발계수
            )
        )

        # 이후에 prim properties를 편집하기 위해 scene에 추가
        # PRACTICE 13: scene에 geometry prim 추가하기

        
        # set collision mesh approximation from geometry prim(visual mesh)
        # PRACTICE 14: collision mesh를 생성하기 위한 object name으로 지정하기
        model_prim = self._scene.get_object(name=f"")
        # none, convexDecomposition, convexHull, boundingSphere, boundingCube, meshSimplification, sdf, sphereFill
        # PRACTICE 15: collision mesh approximation 방법 지정하기​
        model_prim.set_collision_approximation()


        ### fixed joint: 물체 처음 지정한 위치에 고정하기 ###
        # PRACTICE 16: fixed joint 만들어서 물체를 고정하기
        # get stage

        # fixed joint 정의(prim path 활용) - fixed joint는 두 개의 body를 고정시키는 joint

        # Body0: World, Body1: Object
        
        
        # Local position0: World(Body0) 기준으로 joint를 고정할 위치 / rotation
        

        # Local position1: Object(Body1) 기준으로 joint를 고정할 위치 / rotation
        
        
    def set_up_scene(self, scene: Scene) -> None:
        """[summary]
        ShapeNet objects are added to the scene.
        Args:
            scene (Scene): [description]
        """
        # call scene
        self._scene = scene
        # add ground plane
        scene.add_default_ground_plane()

        # basket usd setting
        self.set_usd_background(object_number = 0, object_position = self.background_target_position[0])
        # table usd setting
        self.set_usd_background(object_number = 1, object_position = self.background_target_position[1])

        # add robot to the scene
        self._robot = self.set_robot()
        scene.add(self._robot)

        # add objects on the table
        for i in range(len(self.imported_objects)):
            self.set_usd_objects(object_number = i,
                                object_position = self.objects_position[i],
                                object_orientation = np.random.rand(4))    # random object orientation

        # add camera to the scene
        self.set_camera()
        self._move_task_objects_to_their_frame()
        return
    
    def set_up_scene(self, scene: Scene) -> None:
        """[summary]
        ShapeNet objects are added to the scene.
        Args:
            scene (Scene): [description]
        """
        # call scene
        self._scene = scene
        # add ground plane
        scene.add_default_ground_plane()

        # PRACTICE 17: background object 추가하기
        # basket usd setting

        # table usd setting


        # add robot to the scene
        self._robot = self.set_robot()
        scene.add(self._robot)

        # add objects on the table
        if self.imported_objects is None:   # add only cube
            self._task_object = scene.add(self._objects)
        else:
            # PRACTICE 18: 테이블에 올려둘 objects 추가하기
            for i in range(len(self.imported_objects)):


        # add camera to the scene
        self.set_camera()
        self._move_task_objects_to_their_frame()
        return


    def set_robot(self) -> UR5eHandeye:
        """[summary]

        Returns:
            UR5e: [description]
        """
        working_dir = os.path.dirname(os.path.realpath(__file__))   # same directory with this code
        ur5e_usd_path = os.path.join(working_dir, "../utils/assets/ur5e_handeye_gripper.usd")
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
                           usd_path = ur5e_usd_path,
                           position=self.robot_position)
    

    def set_usd_objects(self, object_number: int, object_position: np.ndarray, object_orientation: np.ndarray) -> None:
        # https://forums.developer.nvidia.com/t/set-mass-and-physicalmaterial-properties-to-prim/229727/4
        
        # define prim for imported objects
        define_prim(self.imported_objects_prim_path + f"_{object_number}")
        define_prim(self.imported_objects_prim_path + f"_{object_number}/model_normalized")


        # task: add usd object to the scene
        # PRACTICE 19: usd path, prim path 지정하기
        self._task_object = add_reference_to_stage(usd_path = ,
                                                   prim_path = )
        
        # rigid body to the scene: mass, velocity, force, etc.
        # PRACTICE 20: prim path, name 지정하기
        rigid_prim = RigidPrim(prim_path = ,
                               position = object_position,
                               orientation = object_orientation,
                               name = ,
                               scale = np.array([0.4] * 3),
                               mass = 1.0,
                               density = 100.0,
        )
        rigid_prim.enable_rigid_body_physics()

        # add to the scene
        self._scene.add(rigid_prim)

        # add geometry prim to the scene: visualizing the object, collision enabled
        # PRACTICE 21: prim path, name 지정하기
        geometry_prim = GeometryPrim(prim_path = ,
                                     name = ,
                                     position = object_position,
                                     orientation = object_orientation,
                                     scale = np.array([0.4] * 3),
                                     collision = True,
                                    )
        # set physics material: surface properties
        # PRACTICE 22: prim path 지정하기
        geometry_prim.apply_physics_material(
            PhysicsMaterial(
                prim_path = ,
                static_friction = 50,
                dynamic_friction = 50,
                restitution = 0.01
            )
        )

        # add to the scene for editing the prim's properties later
        self._scene.add(geometry_prim)

        # set collision mesh approximation from geometry prim(visual mesh)
        # PRACTICE 23: object name으로 모델 지정하고, set_collision_approximation 방법 지정하기
        model_prim = self._scene.get_object(name=)
        model_prim.set_collision_approximation()

    
    def get_params(self) -> dict:
        params_representation = dict()
        
        # The objects on the table
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
        
        # The background objects on the table
        for i in range(len(self.background_objects_list)):          
        # PRACTICE 25: get_params method에서 self.imported_objects 의 코드를 참고하여 self.background_objects_list 의 코드 작성하기



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
        
        for i in range(len(self.imported_objects)):
            observation_dict[self.objects_name_list[i]] = {"position": self.objects_position_list[i],
                                                        "orientation": self.objects_orientation_list[i],
                                                        }
        # PRACTICE 26: get_observations method에서 self.imported_objects 의 코드를 참고하여 self.background_objects_list 의 코드 작성하기
        for i in range(len(self.background_objects_list)):
            
        observation_dict[self._robot.name] = {"joint_positions": joints_state.positions,
                                                "end_effector_position": end_effector_position,
                                                }
        return observation_dict
    
    
    def set_camera(self):
        self.camera = Camera(
            prim_path="/World/ur5e/realsense/Depth",
            frequency=20,
            resolution=(1920, 1080),
        )
        
        self.camera.initialize()
        self.camera.add_distance_to_camera_to_frame()
        self.camera.add_instance_segmentation_to_frame()
        self.camera.add_instance_id_segmentation_to_frame()
        self.camera.add_semantic_segmentation_to_frame()
        self.camera.add_bounding_box_2d_loose_to_frame()
        self.camera.add_bounding_box_2d_tight_to_frame()
        self.camera.add_distance_to_image_plane_to_frame() # for get depth image
        self.camera.add_pointcloud_to_frame() # for get point cloud
        
    def get_camera(self):
        return self.camera