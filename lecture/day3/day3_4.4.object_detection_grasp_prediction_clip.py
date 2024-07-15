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
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.core.utils.viewports import set_camera_view
from omni.replicator.core import random_colours
from omni.isaac.core.utils.rotations import rot_matrix_to_quat, quat_to_rot_matrix

import sys, os
from pathlib import Path
import numpy as np
import random
from glob import glob

import torch
import struct
import torchvision
import cv2
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R

import clip

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.controllers.pick_place_controller_robotiq import PickPlaceController
from day3_3_scene_generation_task import UR5ePickPlace
from utils.extension.clip_extension import ClipExtension, ClipExtensionState

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


###################### background object에 대한 정보 획득 #############################
# scene background(table, bin) 생성을 위한 경로 정보 취득
working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
usd_path = os.path.join(working_dir, 'data/scene_generate_usd/background')

# background object path 취득
background_obj_dirs = [os.path.join(usd_path, obj_name) for obj_name in os.listdir(usd_path)]   # basket, table
background_obj_dirs.sort()
background_object_info = {}
for obj_idx, obj_dir in enumerate(background_obj_dirs):
    if 'table' in obj_dir:
        usd_file = os.path.join(obj_dir, 'table.usd')
    elif 'basket' in obj_dir:
        usd_file = os.path.join(obj_dir, 'basket.usd')
    background_object_info[obj_idx] = {
        'name': os.path.basename(obj_dir),
        'usd_file': usd_file,
        'label': obj_idx,
    }

# background object usd file path
background_usds = [background_object_info[i]['usd_file'] for i in range(len(background_obj_dirs))]

# bin, table 위치(place position) 지정
background_target_position = np.array([[-0.1,0.7,0.0],[0.4,0.3,0.0]])


############### Random한 ShapeNet 물체 생성을 포함하는 Task 생성 ######################

# ShapeNet Dataset 물체들에 대한 정보 취득
shapenet_path = os.path.join(working_dir, 'data/scene_generate_usd/ShapeNetCore_TargetObjects_mat')
obj_files = glob(os.path.join(shapenet_path, '*/*/*.usd'))
obj_files.sort()
object_info = {}
for obj_idx, usd_file in enumerate(obj_files): 
    object_info[obj_idx] = {
        'name': os.path.dirname(usd_file).split('/')[-2],
        'usd_file': usd_file,
        'label': obj_idx, 
    }

# 랜덤한 물체에 대한 usd file path 선택
obje_info = random.sample(list(object_info.values()), 4)
objects_usds = [obje_info[i]['usd_file'] for i in range(len(obje_info))]

# Random하게 생성된 물체들의 번호와 카테고리 출력
print("object: {}".format(obje_info[0]['name']))

# 물체를 생성할 위치 지정(너무 멀어지는 경우 로봇이 닿지 않을 수 있음, 물체 사이의 거리가 가까울 경우 충돌이 발생할 수 있음)
objects_position=[]
# 물체의 위치를 적당한 간격을 두고 랜덤하게 지정
for i in range(len(obje_info)):
    x = np.random.uniform(0.15+0.2*(i//2)+0.15, 0.15+0.2*(i//2)+0.2)                # 0.3 ~ 0.35, 0.5 ~ 0.55
    y = np.random.uniform(-0.25+0.35*(i%2)+0.05, -0.25+0.35*(i%2)+0.3-0.05)         # -0.2 ~ -0.25, 0.15 ~ 0.1
    if i==4:
        y = np.random.uniform(-0.15, -0.05)
    objects_position.append(np.array([x, y, 1.3]))

# 물체간 끼임을 방지하기 위한 offset
offset = np.array([0, 0, 0.01])

# 물체를 놓을 위치(place position) 지정
target_position = np.array([0.0, 0.7, 1.5])
target_orientation = np.array([0, 0, 0, 1])

# World 생성
my_world = World(stage_units_in_meters=1.0)

# Task 생성
my_task = UR5ePickPlace(objects_list=objects_usds,
                        objects_position=objects_position,
                        background_objects_list=background_usds,
                        background_target_position=background_target_position,
                        offset=offset,
                        robot_position=np.array([0.0, 0.0, 1.0])
                        )

# World에 Task 추가
my_world.add_task(my_task)
my_world.reset()

########################################################################


############################# Robot controller 생성 #################################

# Task로부터 ur5e와 camera를 획득
task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])
camera = my_task.get_camera()

# PickPlace controller 생성
my_controller = PickPlaceController(
    name="pick_place_controller", 
    gripper=my_ur5e.gripper, 
    robot_articulation=my_ur5e,
    end_effector_initial_height=1.55
)

# robot control(PD control)을 위한 instance 선언
articulation_controller = my_ur5e.get_articulation_controller()

####################################################################################


################### Grasp model and Detection model, CLIP model load #####################

# Grasp, detection model을 위한 device 설정 및 모델 불러오기
device = 'cpu' #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

# Clip model load
clip_model, preprocess = clip.load("ViT-B/32", device=device)

########################################################################

# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

# GUI 상에서 보는 view point의 위치를 지정
eye = np.array([2.0, 2.0, 3.0])
target = np.array([0.0, 0.0, 1.0])
set_camera_view(eye, target, '/OmniverseKit_Persp', viewport)

# 시뮬레이션 앱 실행 후 물체는 찾기 시작하는 step을 선언
find_step = np.inf

# gui 선언
gui = ClipExtension()
gui.on_startup(ext_id=os.path.abspath(__file__))

########################## 생성한 World #################################

# 생성한 world 에서 physics simulation step
while simulation_app.is_running():
    my_world.step(render=True)
    
    # world가 동작하는 동안 작업 수행
    if my_world.is_playing():
        
        # step이 0일때, world와 controller를 reset
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
            
        # gui 창에서 find object를 클릭했을 때를 기준으로 물체 탐지 시작
        if gui.state == ClipExtensionState.find_object:
            # gui 창에서 입력한 text에 대한 정보 획득
            user_text = gui._user_input_text
            
            # gui 창에서 find_object를 클릭한 시점을 기준으로 find_step 변경
            if  my_world.current_time_step_index < find_step:
                find_step = my_world.current_time_step_index
                
        # find_step을 기준으로 물체 탐지 및 파지 시작
        if my_world.current_time_step_index >= find_step:
            if my_world.current_time_step_index == find_step:
                rgb_image = camera.get_rgb()
                depth_image = camera.get_depth()
                pc, _ = depth2pc(depth_image, camera.get_intrinsics_matrix(), rgb_image)
                
############################################################################################                

############################ Detection Model Inference ##################################

                # rgb 이미지를 detection model의 input에 맞게 transform
                image = transforms(rgb_image)
                image = image[:, :-150, :]
                
                # Detection model inference
                with torch.no_grad():
                    predictions = detection_model([image.to(device)])
                
                # Detection model visualize
                idx = 0
                score_thresh = 0.6
                pred = predictions[idx]

                # image를 numpy로 변환
                predict_image = image.permute(1, 2, 0).cpu().numpy()
                predict_image = predict_image.copy()

                # score가 score_thresh보다 높은 인스턴스만 필터링
                score_filter = [i for i in range(len(pred["scores"])) if pred["scores"][i] > score_thresh]
                num_instances = len(score_filter)
                
                # 각 인스턴스에 랜덤 색상 생성
                colours = random_colours(num_instances, num_channels=3)
                colours = colours.astype(float) / 255.0
                colours = colours.tolist()

                # 카테고리 이름을 랜덤 색상과 매핑
                categories_name = [LABEL_TO_SYNSET.get(c, c) for c in categories_name]
                mapping = {i + 1: cat for i, cat in enumerate(categories_name)}
                labels = [SYNSET_TO_LABEL[mapping[label.item()]] for label in pred["labels"]]
                
                # Detection 결과를 rgb 이미지 위에 표시
                # 각 인스턴스에 대한 크롭된 이미지 저장용
                crop_images = []
                # 바운딩 박스 정보 저장용
                bboxes = []
                
                for bb, label, colour in zip(pred["boxes"].cpu().numpy(), labels, colours):
                    maxint = 2 ** (struct.Struct("i").size * 8 - 1) - 1
                    
                    # 보이지 않는 bounding box는 제외
                    if bb[0] != maxint and bb[1] != maxint:
                        pt1 = (int(bb[0]), int(bb[1]))
                        pt2 = (int(bb[2]), int(bb[3]))
                        
                        # 이미지에 bounding 그리기
                        predict_image = cv2.rectangle(predict_image, pt1, pt2, colour)
                        
                        # 이미지에 레이블 텍스트 그리기
                        predict_image = cv2.putText(predict_image, label, (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, colour)
                        
                        # 해당 바운딩 박스 영역만 크롭하여 저장
                        crop_image = image[:, int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                        crop_images.append(crop_image)
                        bboxes.append(bb)
                        
                # BGR 이미지를 RGB로 변환        
                predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB)
                
                # Detection 결과 cv2로 시각화
                cv2.imshow('detection prediection', predict_image)
                while True:
                    if cv2.getWindowProperty('detection prediection', cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if cv2.waitKey(1) == ord('q'):
                        break
                cv2.destroyAllWindows()
                
############################################################################################
                

############################ CLIP Model Inference ##################################

                # 각 crop image 별 input text와의 유사도 저장
                probs =  []
                for crop_image in crop_images:
                    # text를 deep learning 모델에 넣기 위해 token으로 변환
                    text = clip.tokenize([user_text]).to(device)
                    
                    # image를 clip model의 input size로 변환
                    clip_transform = T.Resize((224, 224))
                    crop_image = clip_transform(crop_image)
                    crop_image = crop_image.unsqueeze(0).to(device)
                    
                    # CLIP model inference
                    with torch.no_grad():
                        logits_per_image, logits_per_text = clip_model(crop_image, text)
                        probs.append(logits_per_image.cpu().numpy())
                        
                # crop image 중에서 input text와 가장 유사도가 큰 이미지 선택
                target_obj_idx = np.argmax(np.array(probs))
                target_obj_bbox = bboxes[target_obj_idx]
                target_image = crop_images[target_obj_idx].permute(1, 2, 0).cpu().numpy()
                
                # BGR 이미지를 RGB로 변환 
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
                
                # CLIP model 결과 cv2로 시각화
                cv2.imshow('target_object', target_image)
                while True:
                    if cv2.getWindowProperty('target_object', cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if cv2.waitKey(1) == ord('q'):
                        break
                cv2.destroyAllWindows()
############################################################################################

                
################################### Grasp Prediction Inference ##############################
                
                # CLIP model에서 추출한 target object bounding box를 이용하여 pointcloud를 crop
                world_center_x, world_center_y = get_world_center(depth_image, camera.get_intrinsics_matrix(), target_obj_bbox)

                if pc is not None:
                    # target object가 있는 부분의 point cloud crop
                    pc = pc[pc[:, 0] > world_center_x-0.15]
                    pc = pc[pc[:, 0] < world_center_x+0.15]
                    pc = pc[pc[:, 1] > world_center_y-0.15]
                    pc = pc[pc[:, 1] < world_center_y+0.15]
                    pc = pc[pc[:, 2] > 0.4]
                    
                    if my_world.current_time_step_index == find_step:
                        # Crop한 pointcloud를 시각화
                        import open3d as o3d
                        pc_o3d = o3d.geometry.PointCloud()
                        pc_o3d.points = o3d.utility.Vector3dVector(pc)
                        o3d.visualization.draw_geometries([pc_o3d])
                        
                        # Grasp prediction model inference, rotation, translation, pre-grasp-translate, width 추출
                        rot, trans, pre_trans, width = inference_cgnet(pc, grasp_model, device)
                        pre_rot =rot.copy()
                        
                        # 예측한 grasp 자세는 카메라 기준이기 때문에, base(World) 기준으로 변환
                        # 1. camera에서 획득한 grasp 자세
                        T_cam_grasp = np.eye(4)
                        T_cam_grasp[:3, :3] = rot[0]
                        T_cam_grasp[:3, 3] = trans[0]
                        
                        # 2. 학습시와 현재 시뮬레이션의 카메라 좌표계가 다르기 때문에 y축으로 90도 만큼 회전시키는 transformation matrix
                        r = R.from_euler('y', 90, degrees=True)
                        T_adjust = np.eye(4)
                        T_adjust[:3, :3] = r.as_matrix()
                        T_cam_grasp = T_cam_grasp @ T_adjust
                        
                        # 3. base에서 부터 camera로의 transformation matrix
                        pose_base_cam, quat_base_cam = camera.get_world_pose(camera_axes='ros') # (3), (4)
                        rot_base_cam = quat_to_rot_matrix(quat_base_cam)
                        T_base_cam = np.eye(4)
                        T_base_cam[:3, :3] = rot_base_cam
                        T_base_cam[:3, 3] = pose_base_cam
                        print('camera transformation matrix in world coordinate: \n', T_base_cam)
                        
                        # 4. base에서 부터 grasp로의 transformation matrix
                        T_base_grasp = T_base_cam @ T_cam_grasp
                        trans = T_base_grasp[:3, 3][np.newaxis, :]
                        rot = T_base_grasp[:3, :3][np.newaxis, :, :]
                        
                        # 만약 z 값이 1.235보다 낮다면, 1.235로 고정(책상과의 충돌 방지)
                        print('trans: ', trans)
                        if trans[0][2] < 1.235:
                            trans[0][2] = 1.235
                            print('z fixed trans: ', trans[0][2])
                        print('rot: ', rot)
                        print('width: ', width[0])
                        
                        # 강건한 파지를 위해 출력된 파지 넓이보다 3cm 더 크게 설정
                        width[0] += -0.03

                        # Cluttered scene에서는 타겟 물체를 바로 잡으러 가면 주변 물체와의 colllision이 발생하기 때문에, 실제 파지할 자세보다 약간 높은 위치에서 pre-grasp를 수행
                        # 좌표 변환은 위와 동일
                        # 1. camera에서 획득한 grasp 자세
                        T_cam_grasp = np.eye(4)
                        T_cam_grasp[:3, :3] = pre_rot[0]
                        T_cam_grasp[:3, 3] = pre_trans[0]
                        
                        # 2. 학습시와 현재 시뮬레이션의 카메라 좌표계가 다르기 때문에 y축으로 90도 만큼 회전시키는 transformation matrix
                        r = R.from_euler('y', 90, degrees=True)
                        T_adjust = np.eye(4)
                        T_adjust[:3, :3] = r.as_matrix()
                        T_cam_grasp = T_cam_grasp @ T_adjust
                        
                        # 3. base에서 부터 camera로의 transformation matrix
                        pose_base_cam, quat_base_cam = camera.get_world_pose(camera_axes='ros')
                        rot_base_cam = quat_to_rot_matrix(quat_base_cam)
                        T_base_cam = np.eye(4)
                        T_base_cam[:3, :3] = rot_base_cam
                        T_base_cam[:3, 3] = pose_base_cam
                        
                        # 4. base에서 부터 grasp로의 transformation matrix
                        T_base_grasp = T_base_cam @ T_cam_grasp
                        pre_trans = T_base_grasp[:3, 3][np.newaxis, :]
                        print('pre trans: ', pre_trans)
                        
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
                
            
# simulation 종료
simulation_app.close()

########################################################################
