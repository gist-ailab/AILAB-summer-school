"""
Default ground plane에 물체들을 원하는 위치에 추가하는 코드입니다.
앞서 배운 내용(day3 - 3.1.add_objects.py)을 바탕으로 scene 생성을 task화 하여 코드 실행합니다.

"""

"""
(물체 불러오기)
1. usd file path를 통해 object(로봇이 잡을 물체)와 background object(테이블,바구니) 불러오기
2. objects position, background object position 정의(object_position_list)
-----------------------------------------------------------------------------------
(World에 Task 추가)
3. World 생성
4. Pink-and-Place scene 만드는 task 생성 -> my_task = UR5ePickPlace(~)
5. World에 Task 추가
-----------------------------------------------------------------------------------
(World run)
6. 생성한 world에서 physics simulation step
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
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport

import sys, os
import numpy as np
import random
from glob import glob

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from day3_3_scene_generation_task import UR5ePickPlace


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
background_target_position = np.array([[0.0,0.8,0.0],[0.5,0.3,0.0]])


############### Random한 ShapeNet 물체 생성을 포함하는 Task 생성 ######################

# ShapeNet Dataset 물체들에 대한 정보 취득
shapenet_dir = os.path.join(working_dir, 'data/scene_generate_usd/ShapeNetCore_TargetObjects_mat')
obj_files = glob(os.path.join(shapenet_dir, '*/*/*.usd'))
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

# World 생성
my_world = World(stage_units_in_meters=1.0)

# Task 생성
my_task = UR5ePickPlace(objects_list = objects_usds,
                        objects_position = objects_position,
                        background_objects_list = background_usds,
                        background_target_position=background_target_position,
                        offset=offset,
                        robot_position=np.array([0.0, 0.0, 1.0])
                        )

# World에 Task 추가
my_world.add_task(my_task)
my_world.reset()

# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/OmniverseKit_Persp')

# GUI 상에서 보는 view point의 위치를 지정
eye = np.array([3.0, 3.0, 4.0])
target = np.array([0.5, 0.5, 1.0])
set_camera_view(eye, target, '/OmniverseKit_Persp', viewport)

########################## 생성한 World #################################
# 생성한 world 에서 physics simulation step
while simulation_app.is_running():
    my_world.step(render=True)
    
    # world가 동작하는 동안 작업 수행
    if my_world.is_playing():
        
        # step이 0일때, world와 controller를 reset
        if my_world.current_time_step_index == 0:
            my_world.reset()
            
# simulation 종료
simulation_app.close()

########################################################################
