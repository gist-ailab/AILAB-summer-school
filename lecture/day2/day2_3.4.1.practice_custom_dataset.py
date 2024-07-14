# 3.3 강의의 코드에 GrdoundTruth 라벨을 추가하여 Dataloader 형태로 구성한 코드
# Dataloader 형태로 구성하면 PyTorch의 DataLoader를 사용하여 데이터를 더 쉽게 사용할 수 있음
# Dataloader로 구현하기 위해 PyTorch의 torch.utils.data.IterableDataset 클래스를 사용하여 무한한 랜덤 씬을 생성하는 데이터셋을 만들 수 있음


# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


"""Dataset with online randomized scene generation for Instance Segmentation training.

Use OmniKit to generate a simple scene. At each iteration, the scene is populated by
adding assets from the user-specified classes with randomized pose and colour. 
The camera position is also randomized before capturing groundtruth consisting of
an RGB rendered image, Tight 2D Bounding Boxes and Instance Segmentation masks. 
"""
import glob
import os
import signal
import sys

import carb
import numpy as np
import torch
from omni.isaac.kit import SimulationApp

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

# Setup default variables
RESOLUTION = (1024, 1024)
OBJ_LOC_MIN = (-50, 5, -50)
OBJ_LOC_MAX = (50, 5, 50)
CAM_LOC_MIN = (100, 0, -100)
CAM_LOC_MAX = (100, 100, 100)
SCALE_MIN = 15
SCALE_MAX = 40

# Default rendering parameters
RENDER_CONFIG = {
    'width': 640,
    'height': 480,
    'headless': False
}


# 데이터셋 클래스 정의
class RandomObjects(torch.utils.data.IterableDataset):
    """Dataset of random ShapeNet objects.
    Objects are randomly chosen from selected categories and are positioned, rotated and coloured
    randomly in an empty room. RGB, BoundingBox2DTight and Instance Segmentation are captured by moving a
    camera aimed at the centre of the scene which is positioned at random at a fixed distance from the centre.

    This dataset is intended for use with ShapeNet but will function with any dataset of USD models
    structured as `root/category/**/*.usd. One note is that this is designed for assets without materials
    attached. This is to avoid requiring to compile MDLs and load textures while training.

    Args:
        categories (tuple of str): Tuple or list of categories. For ShapeNet, these will be the synset IDs.
        max_asset_size (int): Maximum asset file size that will be loaded. This prevents out of memory errors
            due to loading large meshes.
        num_assets_min (int): Minimum number of assets populated in the scene.
        num_assets_max (int): Maximum number of assets populated in the scene.
        split (float): Fraction of the USDs found to use for training.
        train (bool): If true, use the first training split and generate infinite random scenes.
    """

    def __init__(
        self, root, categories, max_asset_size=None, num_assets_min=3, num_assets_max=5, split=0.7, train=True
    ):
        assert len(categories) > 1
        assert (split > 0) and (split <= 1.0)

        # Initialize SimulationAPPs
        self.kit = SimulationApp(RENDER_CONFIG)
        
        # (optional) Set default nucleus path for docker user
        import carb
        import omni.isaac.core.utils.carb as carb_utils
        settings = carb.settings.get_settings()
        carb_utils.set_carb_setting(settings, "/persistent/isaac/asset_root/default", "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1")

        import omni.replicator.core as rep
        import warp as wp
        from omni.kit.viewport.utility import get_active_viewport

        self.rep = rep
        self.wp = wp

        # isaac-sim nucleus에서 assets을 불러오기 위해 assets root path 가져오기 위한 함수
        from omni.isaac.core.utils.nucleus import get_assets_root_path

        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # ShapeNet 카테고리가 이름으로 지정된 경우, synset ID로 변환
        category_ids = [LABEL_TO_SYNSET.get(c, c) for c in categories]
        self.categories = category_ids
        self.range_num_assets = (num_assets_min, max(num_assets_min, num_assets_max))
        try:
            self.references = self._find_usd_assets(root, category_ids, max_asset_size, split, train)
        except ValueError as err:
            carb.log_error(str(err))
            self.kit.close()
            sys.exit()

        # Scene, lights, walls, floor, ceiling, camera, 등 설정
        self.setup_scene()

        # 랜덤 씬 생성을 위한 Replicator 그래프 설정
        self.setup_replicator()

        viewport = get_active_viewport()
        viewport.set_active_camera('/Replicator/Camera_Xform/Camera')

        self.cur_idx = 0
        self.exiting = False

        signal.signal(signal.SIGINT, self._handle_exit)

    # Texture 파일 경로 설정
    def _get_textures(self):
        return [
            self.assets_root_path + "/Isaac/Samples/DR/Materials/Textures/checkered.png",
            self.assets_root_path + "/Isaac/Samples/DR/Materials/Textures/marble_tile.png",
            self.assets_root_path + "/Isaac/Samples/DR/Materials/Textures/picture_a.png",
            self.assets_root_path + "/Isaac/Samples/DR/Materials/Textures/picture_b.png",
            self.assets_root_path + "/Isaac/Samples/DR/Materials/Textures/textured_wall.png",
            self.assets_root_path + "/Isaac/Samples/DR/Materials/Textures/checkered_color.png",
        ]

    def _handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        self.exiting = True

    def close(self):
        self.rep.orchestrator.stop()
        self.kit.close()

    #### 구현 예제 함수 ####
    # Scene, lights, walls, floor, ceiling, camera, 등 설정
    def setup_scene(self):
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.core.utils.stage import set_stage_up_axis

        """Setup lights, walls, floor, ceiling and camera"""
        # Set stage up axis to Y-up
        set_stage_up_axis("y")

        ########################################################################


        ####################### 시뮬레이션 씬(배경) 생성 ############################

        # 배경 설정 (바닥: Cylinder, 천장: Sphere)

        # 카메라 생성

        # annotator 생성 (RGB, BoundingBox2DTight, Instance Segmentation Groundtruth)
        
        ########################################################################

        self.kit.update()

    # ShapeNet의 USD 파일을 찾는 함수
    def _find_usd_assets(self, root, categories, max_asset_size, split, train=True):
        """Look for USD files under root/category for each category specified.
        For each category, generate a list of all USD files found and select
        assets up to split * len(num_assets) if `train=True`, otherwise select the
        remainder.
        """
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
        return references

    #### 구현 예제 함수 ####
    # ShapeNet의 USD 파일을 랜덤하게 씬에 추가하는 함수 (setup_replicaor 함수에서 활용)
    def _instantiate_category(self, category, references):
        ########################################################################


        ####################### ShapeNet USD 랜덤 생성 ###########################

        # Replicator를 활용해 USD 불러오기

        # Replicator를 활용해 visibility 랜덤 설정

        # Replicator를 활용해 클래스 설정

        # Replicator distribution을 활용해 랜덤 포즈(위치, 회전, 스케일) 설정

        # Replicator를 활용해 텍스쳐 랜덤 설정
        
        ########################################################################
        
        return

    #### 구현 예제 함수 ####
    # 랜덤 씬 생성을 위한 Replicator 그래프 설정
    def setup_replicator(self):
        """Setup the replicator graph with various attributes."""

        ########################################################################


        ####################### Replicator 그래프 설정 ###########################

        # 조명 생성

        # Replicator를 활용해 빛 색상 랜덤 설정

        # Replicator를 활용해 카메라 위치 랜덤 설정

        # Replicator를 활용해 에셋 랜덤 생성

        ########################################################################

        # Run replicator for a single iteration without triggering any writes
        self.rep.orchestrator.preview()

    def __iter__(self):
        return self

    #### 구현 예제 함수 ####
    # Dataloader 형태로 구현하기 위해 __next__ 함수를 구현
    def __next__(self):
        ########################################################################


        ################# Image와 Target 반환하는 함수 구현 ###################

        # Replicator orchestrator 활용하여 랜덤 씬 생성

        # RGB, BoundingBox2DTight, Instance Segmentation Groundtruth 획득

        # RGB image 딥러닝 모델에 맞게 변환

        # Bounding Box, 클래스 라벨, Mask 딥러닝 모델에 맞게 변환

        # target에 dict 형태로 groundtruth 저장

        ########################################################################

        return image, target


if __name__ == "__main__":
    "Typical usage"
    import argparse
    import struct

    import cv2

    parser = argparse.ArgumentParser("Dataset test")
    parser.add_argument(
        "--max_asset_size",
        type=float,
        default=10.0,
        help="Maximum asset size to use in MB. Larger assets will be skipped.",
    )
    parser.add_argument(
        "--num_test_images", type=int, default=10, help="number of test images to generate when executing main"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/scene_generate_usd/ShapeNetCore_TargetObjects_mat'),
        help="Root directory containing USDs. If not specified, use {SHAPENET_LOCAL_DIR}_mat as root.",
    )
    args, unknown_args = parser.parse_known_args()

    # If root is not specified use the environment variable SHAPENET_LOCAL_DIR with the _mat suffix as root
    if args.root is None:
        if "SHAPENET_LOCAL_DIR" in os.environ:
            shapenet_local_dir = f"{os.path.abspath(os.environ['SHAPENET_LOCAL_DIR'])}_mat"
            if os.path.exists(shapenet_local_dir):
                args.root = shapenet_local_dir
        if args.root is None:
            print(
                "root argument not specified and SHAPENET_LOCAL_DIR environment variable was not set or the path did not exist"
            )
            exit()

    categories_name = ("bottle", "camera", "car", "earphone", "watercraft")
    dataset = RandomObjects(args.root, categories_name, max_asset_size=args.max_asset_size)
    from omni.replicator.core import random_colours

    categories = [LABEL_TO_SYNSET.get(c, c) for c in categories_name]

    # 예제 이미지를 저장할 디렉토리 생성
    out_dir = os.path.join(os.getcwd(), "_out_gen_imgs", "")
    os.makedirs(out_dir, exist_ok=True)

    image_num = 0
    # Dataset에서 이미지를 가져와 시각화하는 Loop
    for image, target in dataset:
        
        np_image = image.permute(1, 2, 0).cpu().numpy()

        num_instances = len(target["boxes"])
        # Create random colors for each instance as rgb float lists
        colours = random_colours(num_instances, num_channels=3)
        colours = colours.astype(float) / 255.0
        colours = colours.tolist()

        # overlay = np.zeros_like(np_image)
        overlay = np_image.copy()
        for mask, colour in zip(target["masks"].cpu().numpy(), colours):
            overlay[mask, :3] = colour

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

        vis_image = np.hstack((np_image, overlay))
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", vis_image)
        cv2.waitKey(1)
        fig_name = os.path.join(out_dir, f"domain_randomization_test_image_{image_num}.png")
        vis_image = vis_image * 255
        cv2.imwrite(fig_name, vis_image)
        image_num += 1
        if dataset.exiting or (image_num >= args.num_test_images):
            break

    # cleanup
    dataset.close()
