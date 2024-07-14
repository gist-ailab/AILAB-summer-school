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
    
# 랜덤한 물체에 대한 usd file path 선택

# Random하게 생성된 물체들의 ​번호와 카테고리 출력 

# 물체를 생성할 초기 위치(objects_position) 및 offset 설정

# 물체를 놓을 위치(place position) 지정

# World 생성

# Task 생성

# World에 Task 추가


####################################################################################

# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

# 생성한 world 에서 physics simulation step​
while simulation_app.is_running():
    my_world.step(render=True)
    
simulation_app.close()
print("Simulation is Closed")
