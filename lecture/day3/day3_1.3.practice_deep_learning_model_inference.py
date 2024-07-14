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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Grasp 모델 config를 불러오기 위한 경로 설정
current_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_path, 'cgnet/configs/config.yaml')
config = cfg_from_yaml_file(config_path)
# Grasp model을 위한 모델 불러오기
grasp_model = builder.model_builder(config.model)
grasp_model_path = os.path.join(os.path.dirname(current_path), 'data/checkpoint/contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth')
builder.load_model(grasp_model, grasp_model_path)
grasp_model.to(device)
grasp_model.eval()

# Detection model을 위한 모델 불러오기
detection_model_path = os.path.join(os.path.dirname(current_path), 'data/checkpoint/faster_r-cnn_ckpt/best_model.pth')
categories_name = ("bottle", "camera", "car", "earphone", "watercraft")
detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=1 + len(categories_name))
detection_model.load_state_dict(torch.load(detection_model_path))
detection_model.to(device)
detection_model.eval()

# Detection model input을 맞춰주기 위한 transform 생성
transforms = T.Compose([
    T.ToTensor(),
])

####################################################################################

# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

# 시뮬레이션 앱 실행 후 dalay를 위한 변수
max_step = 100

# 생성한 world 에서 physics simulation step​
while simulation_app.is_running():
    
    my_world.step(render=True)
    if my_world.is_playing():
        # step이 0일때, world와 controller를 reset
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
        # 물체가 소환된 후 안정화를 위해 max_step만큼 기다린 후 rgb, depth, pointcloud를 획득
        if my_world.current_time_step_index >= max_step:
            if my_world.current_time_step_index == max_step:
                rgb_image = camera.get_rgb()
                depth_image = camera.get_depth()
                pc, _ = depth2pc(depth_image, camera.get_intrinsics_matrix(), rgb_image)
                
                
############################################################################################                
                
                
############################ Detection Model Inference ##################################
                # rgb 이미지를 detection model의 input에 맞게 transform
                
                # Detection model inference
                
                # Detection model visualize
                idx = 0
                score_thresh = 0.5
                pred = predictions[idx]
                
                # image를 numpy로 변환

                # score가 score_thresh보다 높은 인스턴스만 필터링
                
                # 각 인스턴스에 랜덤 색상 생성
                
                # 카테고리 이름을 랜덤 색상과 매핑
                
                # Detection 결과를 rgb 이미지 위에 표시
                # 각 인스턴스에 대한 크롭된 이미지 저장용

                # 바운딩 박스 정보 저장용

                for bb, label, colour in zip(pred["boxes"].cpu().numpy(), labels, colours):
                    maxint = 2 ** (struct.Struct("i").size * 8 - 1) - 1
                    # 보이지 않는 bounding box는 제외
                    if bb[0] != maxint and bb[1] != maxint:
                        x = bb[0]
                        y = bb[1]
                        w = bb[2] - x
                        h = bb[3] - y
                        
                        # 이미지에 bounding 그리기

                        # 이미지에 레이블 텍스트 그리기

                        # 해당 바운딩 박스 영역만 크롭하여 저장

                # BGR 이미지를 RGB로 변환        

                # Detection 결과 cv2로 시각화

############################################################################################
                
                
################################### Grasp Prediction Inference ##############################
                
                center_x, center_y = int((bb[0]+bb[2])/2), int((bb[1]+bb[3])/2)
                # Detection model에서 추출한 bounding box를 이용하여 pointcloud를 crop
                
                if pc is not None:
                    
                    
                    
                    if my_world.current_time_step_index == max_step:
                        # Crop한 pointcloud를 시각화
                        import open3d as o3d
                        pc_o3d = o3d.geometry.PointCloud()
                        pc_o3d.points = o3d.utility.Vector3dVector(pc)
                        o3d.visualization.draw_geometries([pc_o3d])
                        
                        # Grasp prediction model inference, rotation, translation, pre-grasp-translate, width 추출
                        
                        pre_rot =rot.copy()
                        
                        # 예측한 grasp 자세는 카메라 기준이기 때문에, base(World) 기준으로 변환
                        # 1. camera에서 획득한 grasp 자세
                        T_cam_grasp = np.eye(4)

                        # 2. 학습시와 현재 시뮬레이션의 카메라 좌표계가 다르기 때문에 y축으로 90도 만큼 회전시키는 transformation matrix
                        T_adjust = np.eye(4)
                        
                        # 3. base에서 부터 camera로의 transformation matrix
                        T_base_cam = np.eye(4)
                        print('camera transformation matrix in world coordinate: \n', T_base_cam)
                        
                        # 4. base에서 부터 grasp로의 transformation matrix
                        trans = T_base_grasp[:3, 3][np.newaxis, :]
                        rot = T_base_grasp[:3, :3][np.newaxis, :, :]
                        
                        # 만약 z 값이 0.235보다 낮다면, 0.235로 고정(바닥과의 충돌 방지)

                        # 강건한 파지를 위해 출력된 파지 넓이보다 3cm 더 크게 설정


                        # Cluttered scene에서는 타겟 물체를 바로 잡으러 가면 주변 물체와의 colllision이 발생하기 때문에, 실제 파지할 자세보다 약간 높은 위치에서 pre-grasp를 수행
                        # 좌표 변환은 위와 동일(pre_rot, pre_trans 이용)
                        # 1. camera에서 획득한 grasp 자세

                        # 2. 학습시와 현재 시뮬레이션의 카메라 좌표계가 다르기 때문에 y축으로 90도 만큼 회전시키는 transformation matrix

                        
                        # 3. base에서 부터 camera로의 transformation matrix

                        
                        # 4. base에서 부터 grasp로의 transformation matrix

                        
############################################################################################
                
                
            # 위에서 계산한 파지 자세를 이용하여 파지 및 놓기 수행    
            actions = my_controller.forward(
                picking_position=np.array([trans[0][0], trans[0][1], trans[0][2]]),
                pre_picking_position=np.array([pre_trans[0][0], pre_trans[0][1], pre_trans[0][2]]),
                placing_position=target_position,
                current_joint_positions=my_ur5e.get_joint_positions(),
                end_effector_orientation = rot_matrix_to_quat(rot[0]),
                gripper_width=width[0],
            )
            if my_controller.is_done():
                print("done picking and placing")
                break
            
            articulation_controller.apply_action(actions)
            
        

simulation_app.close()
print("Simulation is Closed")
