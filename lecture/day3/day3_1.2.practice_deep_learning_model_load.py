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
from omni.isaac.core.utils.rotations import rot_matrix_to_quat, quat_to_rot_matrix
from omni.kit.viewport.utility import get_active_viewport
from omni.replicator.core import random_colours

import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import cv2
from glob import glob

import struct
import torchvision
from torchvision import transforms as T

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.controllers.pick_place_controller_robotiq import PickPlaceController
from utils.tasks.pick_place_task import UR5ePickPlace

from cgnet.utils.config import cfg_from_yaml_file
from cgnet.tools import builder
from cgnet.inference_cgnet import inference_cgnet

LABEL_TO_SYNSET = {
    "watercraft": "04530566",
    "camera": "02942699",
    "car": "02958343",
    "bottle": "02876657",
    "earphone": "03261776"
}

SYNSET_TO_LABEL = {v: k for k, v in LABEL_TO_SYNSET.items()}

# change depth image to pointcloud
def depth2pc(depth, K, rgb=None):
    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32)-K[0,2])
    normalized_y = (y.astype(np.float32)-K[1,2])
    
    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]
    
    if rgb is not None:
        rgb = rgb[y, x]
    
    pc = np.vstack([world_x, world_y, world_z]).T
    return (pc, rgb)

# change bounding box to point cloud coordinate
def get_world_center(depth, K, bb): 
    
    x_min, x_max = bb[0], bb[2]
    y_min, y_max = bb[1], bb[3]
    
    if y_min < 0:
        y_min = 0
    if y_max >= 1080:
        y_max = 1079
    if x_min < 0:
        x_min = 0
    if x_max >=1920:
        x_max = 1919
    z_min, z_max = depth[int(y_min), int(x_min)], depth[int(y_max), int(x_max)]
    
    # Convert depth points to 3D
    def to_world(x, y, z):
        world_x = (x - K[0, 2]) * z / K[0, 0]
        world_y = (y - K[1, 2]) * z / K[1, 1]
        return world_x, world_y, z
    
    x_min_w, y_min_w, z_min_w = to_world(x_min, y_min, z_min)
    x_max_w, y_max_w, z_max_w = to_world(x_max, y_max, z_max)
    
    world_center_x = (x_min_w + x_max_w) / 2
    world_center_y = (y_min_w + y_max_w) / 2
    
    return world_center_x, world_center_y

####################################################################################


#################### Shapenet 물체 생성을 포함하는 Task 생성 및 설정 ######################
    
# Shapenet Dataset 물체들에 대한 정보 취득
working_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(Path(working_dir).parent, 'data/scene_generate_usd/ShapeNetCore_TargetObjects_mat')
objs = glob(os.path.join(data_path, '*/*/*.usd'))
object_info = {}
total_object_num = len(objs)
for obj_idx, obj in enumerate(objs):
    object_info[obj_idx] = {
        'name': obj.split('/')[-2],
        'usd_file': obj,
        'label': obj_idx,
    }
    
    
# 랜덤한 물체에 대한 usd file path 선택
objects_list = random.sample(list(object_info.values()), 1)
objects_usd_list = []
for obj_info in objects_list:
    objects_usd_list.append(obj_info['usd_file'])

# Random하게 생성된 물체들의 ​번호와 카테고리 출력 
for i in range(len(objects_list)):
    print("object_{}: {}".format(i, objects_list[i]['name']))

# 물체를 생성할 초기 위치(objects_position) 및 offset 설정
objects_position = np.array([[0.4, 0.35, 0.2]])
offset = np.array([0, 0, 0.1])

# 물체를 놓을 위치(place position) 지정
target_position = np.array([0.5, 0.0, 0.3])
target_orientation = np.array([0, 0, 0, 1])

# World 생성
my_world = World(stage_units_in_meters=1.0)

# Task 생성
my_task = UR5ePickPlace(objects_list = objects_usd_list,
                        objects_position = objects_position,
                        target_position = target_position,
                        offset=offset)

# World에 Task 추가
my_world.add_task(my_task)
my_world.reset()

####################################################################################


############################# Robot controller 생성 #################################

# Task로부터 ur5e와 camera를 획득
task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])
camera = my_task.get_camera()

# PickPlace controller 생성
my_controller = PickPlaceController(
    name="pick_place_controller", 
    gripper=my_ur5e.gripper, 
    robot_articulation=my_ur5e
)

# robot control(PD control)을 위한 instance 선언
articulation_controller = my_ur5e.get_articulation_controller()

####################################################################################


###################### grasp model and Detection model load ########################

# Grasp, detection model을 위한 device 설정 및 모델 불러오기

# Grasp 모델 config를 불러오기 위한 경로 설정

# Grasp model을 위한 모델 불러오기

# Detection model을 위한 모델 불러오기

# Detection model input을 맞춰주기 위한 transform 생성

####################################################################################