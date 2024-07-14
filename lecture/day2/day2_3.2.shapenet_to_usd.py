# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
IsaacSim에서 제공하는 shapenet 데이터셋을 USD로 변환하는 코드입니다.
먼저 ShapeNetCore.v2 데이터셋을 다운로드 받고, 
아래와 같이 SHAPENET_LOCAL_DIR 환경변수를 설정해야 합니다.

>>> export SHAPENET_LOCAL_DIR=/path/to/ShapeNetCore.v2

이후 아래와 같이 실행하면, can, bottle, bowl 카테고리의 100개의 모델을 USD로 변환합니다.

>>> python AILAB-summer-school/lecture/day2/3.2.shapenet_to_usd.py --categories can bottle bowl --max_models 100 --load_materials

변환 가능한 카테고리는 다음과 같습니다.
(table, monitor, phone, watercraft, chair, lamp, speaker, bench, plane, bathtub,
bookcase, bag, basket, bowl, bus, cabinet, camera, car, dishwasher, file, knife,
laptop, mailbox, microwave, piano, pillow, pistol, printer, rocket, sofa, washer,
rifle, can, bottle, bowl, earphone, mug)

"""


import argparse
import os

import carb
from omni.isaac.kit import SimulationApp

if "SHAPENET_LOCAL_DIR" not in os.environ:
    carb.log_error("SHAPENET_LOCAL_DIR not defined:")
    carb.log_error(
        "Please specify the SHAPENET_LOCAL_DIR environment variable to the location of your local shapenet database, exiting"
    )
    exit()

kit = SimulationApp()

from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.shapenet")

from omni.isaac.shapenet import utils

parser = argparse.ArgumentParser("Convert ShapeNet assets to USD")
parser.add_argument(
    "--categories", type=str, nargs="+", default=None, help="List of ShapeNet categories to convert (space seperated)."
)
parser.add_argument(
    "--max_models", type=int, default=50, help="If specified, convert up to `max_models` per category, default is 50"
)
parser.add_argument(
    "--load_materials", action="store_true", help="If specified, materials will be loaded from shapenet meshes"
)
args, unknown_args = parser.parse_known_args()

# Ensure Omniverse Kit is launched via SimulationApp before shapenet_convert() is called
utils.shapenet_convert(args.categories, args.max_models, args.load_materials)
# cleanup
kit.close()
