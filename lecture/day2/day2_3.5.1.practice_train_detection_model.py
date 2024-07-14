# 3.4 강의의 Dataset 클래스를 이용하여 Mask-RCNN 모델을 학습하는 코드


# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


"""Instance Segmentation Training Demonstration

Use a PyTorch dataloader together with OmniKit to generate scenes and groundtruth to
train a [Mask-RCNN](https://arxiv.org/abs/1703.06870) model.
"""

import glob
import os
import signal
import sys

import cv2
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
# RESOLUTION = (1024, 1024)
RESOLUTION = (1920, 1080)
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

        self.kit = SimulationApp(RENDER_CONFIG)

        import omni.replicator.core as rep
        import warp as wp

        self.rep = rep
        self.wp = wp

        # isaac-sim nucleus에서 assets을 불러오기 위해 assets root path 가져오기 위한 함수
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.kit.viewport.utility import get_active_viewport

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

    # Scene, lights, walls, floor, ceiling, camera, 등 설정
    def setup_scene(self):
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.core.utils.stage import set_stage_up_axis

        """Setup lights, walls, floor, ceiling and camera"""
        # Set stage up axis to Y-up
        set_stage_up_axis("y")

        # In a practical setting, the room parameters should attempt to match those of the
        # target domain. Here, we instead opt for simplicity.
        create_prim("/World/Room", "Sphere", attributes={"radius": 1e3, "primvars:displayColor": [(1.0, 1.0, 1.0)]})
        create_prim(
            "/World/Ground",
            "Cylinder",
            position=np.array([0.0, -0.5, 0.0]),
            orientation=euler_angles_to_quat(np.array([90.0, 0.0, 0.0]), degrees=True),
            attributes={"height": 1, "radius": 1e4, "primvars:displayColor": [(1.0, 1.0, 1.0)]},
        )
        create_prim("/World/Asset", "Xform")

        self.camera = self.rep.create.camera()
        self.render_product = self.rep.create.render_product(self.camera, RESOLUTION)

        # Setup annotators that will report groundtruth
        self.rgb = self.rep.AnnotatorRegistry.get_annotator("rgb")
        self.bbox_2d_tight = self.rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        self.instance_seg = self.rep.AnnotatorRegistry.get_annotator("instance_segmentation")
        self.rgb.attach(self.render_product)
        self.bbox_2d_tight.attach(self.render_product)
        self.instance_seg.attach(self.render_product)

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

    # ShapeNet의 USD 파일을 랜덤하게 씬에 추가하는 함수
    def _instantiate_category(self, category, references, ratio=0.6):
        with self.rep.randomizer.instantiate(references, size=1, mode="reference"):
            self.rep.modify.visibility(self.rep.distribution.choice([True, False], [ratio, 1 - ratio]))
            self.rep.modify.semantics([("class", category)])
            self.rep.modify.pose(
                position=self.rep.distribution.uniform(OBJ_LOC_MIN, OBJ_LOC_MAX),
                rotation=self.rep.distribution.uniform((0, -180, 0), (0, 180, 0)),
                scale=self.rep.distribution.uniform(SCALE_MIN, SCALE_MAX),
            )
            self.rep.randomizer.texture(self._get_textures(), project_uvw=True)

    # 랜덤 씬 생성을 위한 Replicator 그래프 설정
    def setup_replicator(self):
        """Setup the replicator graph with various attributes."""

        # Create two sphere lights
        light1 = self.rep.create.light(light_type="sphere", position=(-450, 350, 350), scale=100, intensity=30000.0)
        light2 = self.rep.create.light(light_type="sphere", position=(450, 350, 350), scale=100, intensity=30000.0)

        with self.rep.new_layer():
            with self.rep.trigger.on_frame():
                # Randomize light colors
                with self.rep.create.group([light1, light2]):
                    self.rep.modify.attribute("color", self.rep.distribution.uniform((0.1, 0.1, 0.1), (1.0, 1.0, 1.0)))

                # Randomize camera position
                with self.camera:
                    self.rep.modify.pose(
                        position=self.rep.distribution.uniform(CAM_LOC_MIN, CAM_LOC_MAX), look_at=(0, 0, 0)
                    )

                # Randomize asset positions and textures
                for category, references in self.references.items():
                    self._instantiate_category(category, references)

        # Run replicator for a single iteration without triggering any writes
        self.rep.orchestrator.preview()

    def __iter__(self):
        return self

    # Dataloader 형태로 구현하기 위해 __next__ 함수를 구현
    def __next__(self):
        # Step - trigger a randomization and a render
        self.rep.orchestrator.step(rt_subframes=4)

        # Collect Groundtruth
        gt = {
            "rgb": self.rgb.get_data(device="cuda"),
            "boundingBox2DTight": self.bbox_2d_tight.get_data(device="cpu"),
            "instanceSegmentation": self.instance_seg.get_data(device="cuda"),
        }

        # RGB
        # Drop alpha channel
        image = self.wp.to_torch(gt["rgb"])[..., :3]

        # Normalize between 0. and 1. and change order to channel-first.
        image = image.float() / 255.0
        image = image.permute(2, 0, 1)

        # Bounding Box
        gt_bbox = gt["boundingBox2DTight"]["data"]

        # Check if there are no bounding boxes
        if len(gt_bbox) == 0:
            target = {
                "boxes": torch.empty((0, 4), device="cuda"),
                "labels": torch.empty((0,), dtype=torch.int64, device="cuda"),
                "masks": torch.empty((0, image.shape[1], image.shape[2]), dtype=bool, device="cuda"),
                "image_id": torch.LongTensor([self.cur_idx]),
                "area": torch.empty((0,), device="cuda"),
                "iscrowd": torch.empty((0,), dtype=torch.bool, device="cuda"),
            }
            self.cur_idx += 1
            return image, target

        # Create mapping from categories to index
        bboxes = torch.tensor(gt_bbox[["x_min", "y_min", "x_max", "y_max"]].tolist(), device="cuda")
        id_to_labels = gt["boundingBox2DTight"]["info"]["idToLabels"]
        prim_paths = gt["boundingBox2DTight"]["info"]["primPaths"]

        # For each bounding box, map semantic label to label index
        cat_to_id = {cat: i + 1 for i, cat in enumerate(self.categories)}
        semantic_labels_mapping = {int(k): v.get("class", "") for k, v in id_to_labels.items()}
        semantic_labels = [cat_to_id[semantic_labels_mapping[i]] for i in gt_bbox["semanticId"]]
        labels = torch.tensor(semantic_labels, device="cuda")

        # Calculate bounding box area for each area
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # Identify invalid bounding boxes to filter final output
        valid_areas = (areas > 0.0) * (areas < (image.shape[1] * image.shape[2]))

        # Instance Segmentation
        instance_data = self.wp.to_torch(gt["instanceSegmentation"]["data"].view(self.wp.int32)).squeeze()
        path_to_instance_id = {v: int(k) for k, v in gt["instanceSegmentation"]["info"]["idToLabels"].items()}

        instance_list = [im[0] for im in gt_bbox]
        masks = torch.zeros((len(instance_list), *instance_data.shape), dtype=bool, device="cuda")

        # Filter for the mask of each object
        for i, prim_path in enumerate(prim_paths):
            # Merge child instances of prim_path as one instance
            for instance in path_to_instance_id:
                if prim_path in instance:
                    masks[i] += torch.isin(instance_data, path_to_instance_id[instance])

        target = {
            "boxes": bboxes[valid_areas],
            "labels": labels[valid_areas],
            "masks": masks[valid_areas],
            "image_id": torch.LongTensor([self.cur_idx]),
            "area": areas[valid_areas],
            "iscrowd": torch.BoolTensor([False] * len(bboxes[valid_areas])),  # Assume no crowds
        }

        self.cur_idx += 1
        return image, target


def main(args):
    device = "cuda"

    categories_name = ("bottle", "camera", "car", "earphone", "watercraft")
    train_set = RandomObjects(
        args.root, categories_name, num_assets_min=3, num_assets_max=5, max_asset_size=args.max_asset_size
    )

    def handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        train_set.exiting = True

    signal.signal(signal.SIGINT, handle_exit)

    import struct

    import torch
    import torchvision
    from omni.replicator.core import random_colours
    from torch.utils.data import DataLoader

    # DataLoader를 이용하여 데이터셋을 불러오기
    train_loader = DataLoader(train_set, batch_size=2, collate_fn=lambda x: tuple(zip(*x)))

    # Faster R-CNN 모델을 불러오기
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=1 + len(categories_name))
    
    # Instance Segmentation을 위한 Mask-RCNN 모델을 불러오기
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, num_classes=1 + len(categories_name))
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Directory to save the train images to
    out_dir = os.path.join(os.getcwd(), "_out_train_imgs", "")
    os.makedirs(out_dir, exist_ok=True)

    # Training loop
    for i, train_batch in enumerate(train_loader):
        if i > args.max_iters or train_set.exiting:
            print("Exiting ...")
            train_set.close()
            break

        ########################################################################


        ################### Faster R-CNN 학습하는 코드 구현 ########################

        # train_batch에 저장된 image, target을 이용

        # 모델에 image를 입력하고, 학습을 위한 loss 계산

        # Backpropagation을 통해 모델 학습

        # Checkpoint 저장

        # Evaluation 및 Visualization 코드 구현

        ########################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Dataset test")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing ShapeNet USDs. If not specified, use {SHAPENET_LOCAL_DIR}_nomat as root.",
    )
    # parser.add_argument(
    #     "--categories", type=str, nargs="+", required=True, help="List of ShapeNet categories to use (space seperated)."
    # )
    parser.add_argument(
        "--max_asset_size",
        type=float,
        default=10.0,
        help="Maximum asset size to use in MB. Larger assets will be skipped.",
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_iters", type=float, default=30000, help="Number of training iterations.")
    parser.add_argument("--visualize", action="store_true", help="Visualize predicted masks during training.")
    parser.add_argument("--save_model", action="store_true", help="Save model weights during training.")
    args, unknown_args = parser.parse_known_args()

    # If root is not specified use the environment variable SHAPENET_LOCAL_DIR with the _nomat suffix as root
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

    main(args)
