# Replicator와 ShapeNet 물체를 활용하여 이미지와 GroundTruth를 생성하는 코드입니다.

import glob
import os
import signal
import sys

import carb
import numpy as np
import torch
from omni.isaac.kit import SimulationApp

import argparse
import struct

import cv2

# ShapeNet을 이용하기 위한 환경변수 설정
LABEL_TO_SYNSET = {
    "table": "04379243",
    "monitor": "03211117",
    "phone": "04401088",
    "watercraft": "04530566",
    "chair": "03001627",
    "lamp": "03636649",
    "speaker": "03691459",
    "bench": "02828884",
    "plane": "02691156",
    "bathtub": "02808440",
    "bookcase": "02871439",
    "bag": "02773838",
    "basket": "02801938",
    "bowl": "02880940",
    "bus": "02924116",
    "cabinet": "02933112",
    "camera": "02942699",
    "car": "02958343",
    "dishwasher": "03207941",
    "file": "03337140",
    "knife": "03624134",
    "laptop": "03642806",
    "mailbox": "03710193",
    "microwave": "03761084",
    "piano": "03928116",
    "pillow": "03938244",
    "pistol": "03948459",
    "printer": "04004475",
    "rocket": "04099429",
    "sofa": "04256520",
    "washer": "04554684",
    "rifle": "04090263",
    "can": "02946921",
    "bottle": "02876657",
    "bowl": "02880940",
    "earphone": "03261776",
    "mug": "03797390",
}
SYNSET_TO_LABEL = {v: k for k, v in LABEL_TO_SYNSET.items()}

# ShapeNet 데이터셋 경로 설정
if "SHAPENET_LOCAL_DIR" in os.environ:
    shapenet_local_dir = f"{os.path.abspath(os.environ['SHAPENET_LOCAL_DIR'])}_mat"
    if os.path.exists(shapenet_local_dir):
        root = shapenet_local_dir
else:
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/scene_generate_usd/ShapeNetCore_TargetObjects_mat')
    
# ShapeNet 카테고리가 이름으로 지정된 경우, synset ID로 변환
categories_name = ("bottle", "camera", "car", "earphone", "watercraft")
category_ids = [LABEL_TO_SYNSET.get(c, c) for c in categories_name]
categories = category_ids


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

# 데이터셋 생성 변수 설정
num_test_images = 10
max_asset_size = 10.0
num_assets_min=3
num_assets_max=5
train=True
split=0.7
range_num_assets = (num_assets_min, max(num_assets_min, num_assets_max))

# ShapeNet USD 파일 경로 설정
try:
    references = {}
    for category in categories:
        all_assets = glob.glob(os.path.join(root, category, "*/*.usd"), recursive=True)
        print(os.path.join(root, category, "*/*.usd"))
        # Filter out large files (which can prevent OOM errors during training)
        if max_asset_size is None:
            assets_filtered = all_assets
        else:
            assets_filtered = []
            for a in all_assets:
                if os.stat(a).st_size > max_asset_size * 1e6:
                    print(f"{a} skipped as it exceeded the max size {max_asset_size} MB.")
                else:
                    assets_filtered.append(a)

        num_assets = len(assets_filtered)
        if num_assets == 0:
            raise ValueError(f"No USDs found for category {category} under max size {max_asset_size} MB.")

        if train:
            references[category] = assets_filtered[: int(num_assets * split)]
        else:
            references[category] = assets_filtered[int(num_assets * split) :]
except ValueError as err:
    carb.log_error(str(err))
    kit.close()
    sys.exit()


# Warp: 고성능 시뮬레이션 및 그래픽 코드를 작성하기 위한 Python 프레임워크
import warp as wp
import omni.replicator.core as rep

# isaac-sim nucleus에서 assets을 불러오기 위해 assets root path 가져오기 위한 함수
from omni.isaac.core.utils.nucleus import get_assets_root_path
assets_root_path = get_assets_root_path()

# 기본 환경변수 설정 (해상도, 오브젝트 위치, 카메라 위치, 스케일)
RESOLUTION = (1024, 1024)
OBJ_LOC_MIN = (-50, 5, -50)
OBJ_LOC_MAX = (50, 5, 50)
CAM_LOC_MIN = (100, 0, -100)
CAM_LOC_MAX = (100, 100, 100)
SCALE_MIN = 15
SCALE_MAX = 40

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

# Annotator 설정 (rgb, bounding box, segmentation)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
bbox_2d_tight = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
instance_seg = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
rgb.attach(render_product)
bbox_2d_tight.attach(render_product)
instance_seg.attach(render_product)

kit.update()

# 두 개의 sphere light 생성
light1 = rep.create.light(light_type="sphere", position=(-450, 350, 350), scale=100, intensity=30000.0)
light2 = rep.create.light(light_type="sphere", position=(450, 350, 350), scale=100, intensity=30000.0)

# Isaac-sim nucleus에 내장된 Texture 경로 설정
textures = [
    assets_root_path + "/Isaac/Samples/DR/Materials/Textures/checkered.png",
    assets_root_path + "/Isaac/Samples/DR/Materials/Textures/marble_tile.png",
    assets_root_path + "/Isaac/Samples/DR/Materials/Textures/picture_a.png",
    assets_root_path + "/Isaac/Samples/DR/Materials/Textures/picture_b.png",
    assets_root_path + "/Isaac/Samples/DR/Materials/Textures/textured_wall.png",
    assets_root_path + "/Isaac/Samples/DR/Materials/Textures/checkered_color.png",
]

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
        
        # 랜덤한 카테고리의 에셋을 랜덤한 위치와 텍스쳐 설정
        for category, reference in references.items():
            # rep.randomizer.instantiate(): 에셋 경로를 입력받아 랜덤한 위치에 에셋을 생성
            with rep.randomizer.instantiate(reference, size=1, mode="reference"):
                rep.modify.visibility(rep.distribution.choice([True, False], [0.6, 0.4]))
                rep.modify.semantics([("class", category)])
                rep.modify.pose(
                    position=rep.distribution.uniform(OBJ_LOC_MIN, OBJ_LOC_MAX),
                    rotation=rep.distribution.uniform((0, -180, 0), (0, 180, 0)),
                    scale=rep.distribution.uniform(SCALE_MIN, SCALE_MAX),
                )
                rep.randomizer.texture(textures, project_uvw=True)


# Orchestrator: Replicator graph의 실행을 관리하는 클래스
# Orchestrator를 사용하여 trigerring 없이 한 번 실행
rep.orchestrator.preview()

# GT 시각화를 위한 random_colours 함수
from omni.replicator.core import random_colours

# 예시 이미지 저장할 디렉토리 생성
out_dir = os.path.join(os.getcwd(), "_out_gen_imgs", "")
os.makedirs(out_dir, exist_ok=True)

# 데이터셋 생성
for i in range(num_test_images):
    print(i)
    # Step - trigger a randomization and a render
    rep.orchestrator.step(rt_subframes=4)

    # RGB, Bounding Box, Instance Segmentation 수집
    gt = {
        "rgb": rgb.get_data(device="cuda"),
        "boundingBox2DTight": bbox_2d_tight.get_data(device="cpu"),
        "instanceSegmentation": instance_seg.get_data(device="cuda"),
    }

    # RGB 이미지를 torch tensor로 변환 및 alpha channel 제거
    image = wp.to_torch(gt["rgb"])[..., :3]

    # 이미지 채널 순서 변경
    image = image.permute(2, 0, 1)

    # Bounding Box
    gt_bbox = gt["boundingBox2DTight"]["data"]

    # Bounding Box 정보가 없는 경우, 빈 tensor 생성
    if len(gt_bbox) == 0:
        target = {
            "boxes": torch.empty((0, 4), device="cuda"),
            "labels": torch.empty((0,), dtype=torch.int64, device="cuda"),
            "masks": torch.empty((0, image.shape[1], image.shape[2]), dtype=bool, device="cuda"),
            "image_id": torch.LongTensor([i]),
            "area": torch.empty((0,), device="cuda"),
            "iscrowd": torch.empty((0,), dtype=torch.bool, device="cuda"),
        }
    else:
        # 카테고리를 인덱스로 매핑
        bboxes = torch.tensor(gt_bbox[["x_min", "y_min", "x_max", "y_max"]].tolist(), device="cuda")
        id_to_labels = gt["boundingBox2DTight"]["info"]["idToLabels"]
        prim_paths = gt["boundingBox2DTight"]["info"]["primPaths"]

        # 각 bounding box에 대해 semantic label을 label index로 매핑
        cat_to_id = {cat: i + 1 for i, cat in enumerate(categories)}
        semantic_labels_mapping = {int(k): v.get("class", "") for k, v in id_to_labels.items()}
        semantic_labels = [cat_to_id[semantic_labels_mapping[i]] for i in gt_bbox["semanticId"]]
        labels = torch.tensor(semantic_labels, device="cuda")

        # 각 bounding box의 면적 계산
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # 유효하지 않은 bounding box 식별하여 최종 출력 필터링
        valid_areas = (areas > 0.0) * (areas < (image.shape[1] * image.shape[2]))

        # Instance Segmentation
        instance_data = wp.to_torch(gt["instanceSegmentation"]["data"].view(wp.int32)).squeeze()
        path_to_instance_id = {v: int(k) for k, v in gt["instanceSegmentation"]["info"]["idToLabels"].items()}

        instance_list = [im[0] for im in gt_bbox]
        masks = torch.zeros((len(instance_list), *instance_data.shape), dtype=bool, device="cuda")

        # 각 object의 mask를 필터링
        for i, prim_path in enumerate(prim_paths):
            # prim_path의 모든 하위 인스턴스를 하나의 인스턴스로 병합
            for instance in path_to_instance_id:
                if prim_path in instance:
                    masks[i] += torch.isin(instance_data, path_to_instance_id[instance])

        target = {
                "boxes": bboxes[valid_areas],
                "labels": labels[valid_areas],
                "masks": masks[valid_areas],
                "image_id": torch.LongTensor([i]),
                "area": areas[valid_areas],
                "iscrowd": torch.BoolTensor([False] * len(bboxes[valid_areas])),  # Assume no crowds
            }


    np_image = image.permute(1, 2, 0).cpu().numpy()

    num_instances = len(target["boxes"])
    # 각 instance에 대해 랜덤한 색상 생성
    colours = random_colours(num_instances, num_channels=3)
    colours = colours.tolist()

    # 이미지에 mask overlay
    overlay = np_image.copy()
    for mask, colour in zip(target["masks"].cpu().numpy(), colours):
        overlay[mask, :3] = colour

    # 이미지에 bounding box overlay
    mapping = {i + 1: cat for i, cat in enumerate(categories)}
    labels = [SYNSET_TO_LABEL[mapping[label.item()]] for label in target["labels"]]
    for bb, label, colour in zip(target["boxes"].tolist(), labels, colours):
        maxint = 2 ** (struct.Struct("i").size * 8 - 1) - 1
        # if a bbox is not visible, do not draw
        if bb[0] != maxint and bb[1] != maxint:
            x = bb[0]
            y = bb[1]
            w = bb[2] - x
            h = bb[3] - y
            pt1 = (int(bb[0]), int(bb[1]))
            pt2 = (int(bb[2]), int(bb[3]))
            overlay = cv2.rectangle(overlay, pt1, pt2, colour)
            overlay = cv2.putText(overlay, label, (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, colour)

    # 이미지 출력
    vis_image = np.hstack((np_image, overlay))
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", vis_image)
    cv2.waitKey(1)
    fig_name = os.path.join(out_dir, f"domain_randomization_test_image_{i}.png")
    cv2.imwrite(fig_name, vis_image)
rep.orchestrator.stop()
kit.close()