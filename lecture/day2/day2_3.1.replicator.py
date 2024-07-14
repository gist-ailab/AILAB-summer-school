# Replicator를 이용하여 이미지를 생성하는 코드입니다.

# Replicator
"""Replicator: A framework for developing custom synthetic data generation pipelines and services

Generating synthetic data with Omniverse Replicator is a two step process.

The first step brings assets into the scene, defines and registers randomizers, annotators and writers. 
It also defines the event triggers for randomizers to execute. 
Under the hood this first step builds OmniGraph nodes to execute these steps efficiently. 

Once the OmniGraph nodes are built, the second step executes these nodes to generate the data, 
annotations and writes output to the disk in the desired form. 
Replicator APIs abstract these complexities from the users.
"""
import os
import numpy as np
from omni.isaac.kit import SimulationApp

import cv2


# SimulationApp 설정
RENDER_CONFIG = {
    'width': 640,
    'height': 480,
    'headless': False
}
kit = SimulationApp(RENDER_CONFIG)
import carb
import omni.isaac.core.utils.carb as carb_utils
settings = carb.settings.get_settings()
carb_utils.set_carb_setting(settings, "/persistent/isaac/asset_root/default", "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1")

# Warp: 고성능 시뮬레이션 및 그래픽 코드를 작성하기 위한 Python 프레임워크
import warp as wp
import omni.replicator.core as rep

# 기본 환경변수 설정 (해상도, 오브젝트 위치, 카메라 위치, 스케일)
RESOLUTION = (1024, 1024)
OBJ_LOC_MIN = (-50, 5, -50)
OBJ_LOC_MAX = (50, 5, 50)
CAM_LOC_MIN = (100, 0, -100)
CAM_LOC_MAX = (100, 100, 100)
SCALE_MIN = 5
SCALE_MAX = 15


"""Lights, walls, floor, ceiling and camera 설정"""
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import set_stage_up_axis
from omni.kit.viewport.utility import get_active_viewport

# Y-up으로 stage up axis 설정
set_stage_up_axis("y")

# 배경 세팅 (방: Sphere, 바닥: Cylinder)
create_prim("/World/Room", "Sphere", attributes={"radius": 1e3, "primvars:displayColor": [(1.0, 1.0, 1.0)]})
create_prim(
    "/World/Ground",
    "Cylinder",
    position=np.array([0.0, -0.5, 0.0]),
    orientation=euler_angles_to_quat(np.array([90.0, 0.0, 0.0]), degrees=True),
    attributes={"height": 1, "radius": 1e4, "primvars:displayColor": [(1.0, 1.0, 1.0)]},
)
create_prim("/World/Asset", "Xform")

# 카메라 설정
camera = rep.create.camera()
render_product = rep.create.render_product(camera, RESOLUTION)
viewport = get_active_viewport()
viewport.set_active_camera('/Replicator/Camera_Xform/Camera')

# Annotator 설정 (본 예제에서는 rgb만 설정 / bounding box, segmentation, 등 다양한 annotator 설정 가능)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(render_product)

kit.update()

# 두 개의 sphere light 생성
light1 = rep.create.light(light_type="sphere", position=(-450, 350, 350), scale=100, intensity=30000.0)
light2 = rep.create.light(light_type="sphere", position=(450, 350, 350), scale=100, intensity=30000.0)

# Replicator randomizer graph 설정
with rep.new_layer():
    # Replicator on_frame trigger 세팅, on_frame 안에서 랜덤화된 새로운 프레임 생성
    with rep.trigger.on_frame():
        # Light 색상 랜덤화
        with rep.create.group([light1, light2]):
            rep.modify.attribute("color", rep.distribution.uniform((0.1, 0.1, 0.1), (1.0, 1.0, 1.0)))

        # 카메라 위치 랜덤화
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(CAM_LOC_MIN, CAM_LOC_MAX), look_at=(0, 0, 0)
            )

        # 3개의 큐브를 랜덤한 위치와 색상으로 설정
        for idx in range(3):
            cube = rep.create.cube(
                position=rep.distribution.uniform(OBJ_LOC_MIN, OBJ_LOC_MAX),
                rotation=rep.distribution.uniform((0, -180, 0), (0, 180, 0)),
                scale=rep.distribution.uniform(SCALE_MIN, SCALE_MAX),
                material=rep.create.material_omnipbr(diffuse=rep.distribution.uniform((0.1, 0.1, 0.1), (1, 1, 1))),
            )

# 예시 이미지 저장할 디렉토리 생성
out_dir = os.path.join(os.getcwd(), "_out_gen_imgs_cube", "")
os.makedirs(out_dir, exist_ok=True)


# Orchestrator: Replicator graph의 실행을 관리하는 클래스
# Orchestrator를 사용하여 trigerring 없이 한 번 실행
rep.orchestrator.preview()

# 데이터셋 생성 변수 설정
num_test_images = 10

# 데이터셋 생성
for i in range(num_test_images):
    # Step - trigger a randomization and a render
    rep.orchestrator.step(rt_subframes=4)

    # RGB 수집 (추후에 다른 GT 추가 예정)
    gt = {
        "rgb": rgb.get_data(device="cuda"),
    }

    # RGB 이미지를 torch tensor로 변환 및 alpha channel 제거
    image = wp.to_torch(gt["rgb"])[..., :3]

    # opencv 활용하여 이미지 출력
    np_image = image.cpu().numpy()
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", np_image)
    cv2.waitKey(1)

    # 이미지 저장
    cv2.imwrite(os.path.join(out_dir, f"image_{i}.png"), np_image)


rep.orchestrator.stop()
kit.close()