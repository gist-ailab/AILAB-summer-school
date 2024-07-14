"""
Default ground plane에 물체들을 원하는 위치에 추가하는 코드입니다.
물체를 추가할때 여러가지 특성을 설정할 수 있습니다.

"""

"""
(물체 불러오기)
1. World 생성
2. usd file path를 통해 background object 불러오기(background_usds)
3. background object의 위치 설정(object_position_list)
-----------------------------------------------------------------------------------
(position 및 size를 위한 전처리)
4. background object에 특징을 부여하기 위한 prim path 정의
5. 임의의 RigidPrim을 만들어 background object의 mesh size 측정(get_mesh_size method 활용)
6. mesh size를 통해 모든 물체를 1*1*1로 만들기 위한 scale factor 설정
7. 각 물체 중심이 기준인 position을 mesh size를 통해 아랫면이 0이 되도록 설정
8. 물체를 고정하기 위한 fixed joint position 설정
-----------------------------------------------------------------------------------
(물체 prim 설정)
9. RigidPrim을 통해 물체를 생성하고 scene에 추가
10. GeometryPrim을 통해 물체의 시각화 및 충돌 설정
11. PhysicsMaterial을 통해 물체의 surface properties 설정
12. 물체를 고정하기 위한 fixed joint 설정
13. 물체의 collision mesh를 approximate하기 위한 방법 설정
-----------------------------------------------------------------------------------
(world run)
14. 생성한 world에서 physics simulation step

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
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from omni.isaac.core.materials import PhysicsMaterial

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import define_prim
from pxr import Usd, UsdGeom, Gf, UsdPhysics
import omni.usd

from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport

import numpy as np
import os

# mesh size 측정(x,y,z)
def get_mesh_size(prim_path):
    # 현재 stage 취득
    stage = omni.usd.get_context().get_stage()
    # prim path로 prim 취득
    prim = stage.GetPrimAtPath(prim_path)
    # prim size를 측정하기 위한 bbox cache 생성
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox_cache.Clear()
    # prim의 bbox 취득
    prim_bbox = bbox_cache.ComputeWorldBound(prim)
    prim_range = prim_bbox.ComputeAlignedRange()
    # prim size 취득
    prim_size = prim_range.GetSize()
    return prim_size

# World 생성
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
my_world.reset()

# background objects(table, bin) 생성을 위한 상위 파일경로 취득
working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
usd_path = os.path.join(working_dir, 'data/scene_generate_usd/background')

# background object 경로 취득
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
background_usds = [background_object_info[i]['usd_file'] for i in range(len(background_obj_dirs))]

# background object 순서대로 1개씩 scene에 올리기
for i in range(len(background_usds)):
    # initialize usd_path, obj_name, prim_path
    usd_path = background_usds[i]
    obj_name = usd_path.split('/')[-2]  # table, basket
    background_object_prim_path = "/World/"+obj_name # 1st, 2nd, 3rd object
    
    # define prim for background object
    define_prim(background_object_prim_path)
    define_prim(background_object_prim_path + "/model")

    # add usd object to the scene
    add_reference_to_stage(usd_path = usd_path,
                            prim_path = background_object_prim_path)
    
    # object geometry 값
    if obj_name == "table":
        scale = np.array([3.0, 1.0, 1.5])
        object_position = np.array([0.0,0.0,0.0])
        xzy2xyz = np.array([0.7071, 0.7071, 0, 0])
        offset = np.array([0, 0, 0.01])        # 물체 충돌 방지를 위한 offset 값

    elif obj_name == "basket":
        scale = np.array([0.5, 0.15, 0.5])
        object_position = np.array([-1.0,0.0,0.0])
        xzy2xyz = np.array([0.7071, 0.7071, 0, 0])
        offset = np.array([0, 0, 0.01])         # 물체 충돌 방지를 위한 offset 값

    # initial prim for size check
    RigidPrim(prim_path = background_object_prim_path,
                            position = [0,0,0], 
                            orientation=xzy2xyz,
                            scale = [1,1,1],
                            name = obj_name,
                            mass = 10.0,
                            density = 10.0)
    
    # get mesh size
    mesh_size = get_mesh_size(background_object_prim_path)
    
    # convert scale factor after mesh size normalize (1*1*1)
    if obj_name == "table":
        # 가로,세로,높이 모두 1로 만들기
        scale[0] = scale[0] / mesh_size[0]     # x-axis scale(sim 기준)
        scale[1] = scale[1] / mesh_size[2]     # z-axis scale(sim 기준) - 높이축
        scale[2] = scale[2] / mesh_size[1]     # y-axis scale(sim 기준)
        # 테이블 높이의 중간으로 위치 조정
        object_position[2] = object_position[2] + mesh_size[2] * scale[1] / 2 + offset[2]  # 테이블높이/2
        # 고정할 위치를 물체 위치로 설정
        fixed_joint_postion = Gf.Vec3f(object_position[0],object_position[1],object_position[2])

    elif obj_name == "basket":
        # 가로,세로, 높이 모두 1로 만들기
        scale[0] = scale[0] / mesh_size[0]     # x-axis scale(sim 기준)
        scale[1] = scale[1] / mesh_size[2]    # z-axis scale(sim 기준) - 높이축
        scale[2] = scale[2] / mesh_size[1]     # y-axis scale(sim 기준)
        # 바구니 높이의 중간으로 위치 조정 + 테이블 높이 + offset
        object_position[2] = object_position[2] + mesh_size[2] * scale[1] / 2 + 1.0 + offset[2]
        # 고정할 위치를 물체 위치로 설정
        fixed_joint_postion = Gf.Vec3f(object_position[0],object_position[1],object_position[2])

    # rigid body to the scene: mass, velocity, force, etc.
    rigid_prim = RigidPrim(prim_path = background_object_prim_path,
                            position = object_position+offset,
                            orientation=xzy2xyz,
                            scale = scale,
                            name = obj_name,
                            mass = 10.0,
                            density = 10.0)
    
    # rigid body add to the scene
    my_world.scene.add(rigid_prim)
    rigid_prim.enable_rigid_body_physics()
    
    # add geometry prim to the scene: visualizing the object, collision enabled
    geometry_prim = GeometryPrim(prim_path = background_object_prim_path + "/model",
                                    name = f"{obj_name}_model",
                                    position = object_position+offset, 
                                    orientation=xzy2xyz,
                                    scale = [1,1,1],
                                    collision = True,
                                )
    # set physics material: surface properties
    geometry_prim.apply_physics_material(
        PhysicsMaterial(
            prim_path = background_object_prim_path + "/physics_material",
            static_friction = 50,       # 정적마찰계수
            dynamic_friction = 50,      # 동적마찰계수
            restitution = 0.01          # 반발계수
        )
    )

    # 이후에 prim properties를 편집하기 위해 scene에 추가
    my_world.scene.add(geometry_prim)
    
    # 물체 처음 지정한 위치에 고정하기
    # get stage
    stage = omni.usd.get_context().get_stage()
    # fixed joint 정의(prim path 활용) - fixed joint는 두 개의 body를 고정시키는 joint
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, background_object_prim_path + "/model/fixed_joint")
    # Body0: World, Body1: Object
    fixed_joint.CreateBody0Rel().SetTargets(['/World/defaultGroundPlane/GroundPlane'])
    fixed_joint.CreateBody1Rel().SetTargets([background_object_prim_path + "/model"])
    # fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(-1.0, 0.0, 0.5))
    # Local position0: World(Body0) 기준으로 joint를 고정할 위치 / rotation
    fixed_joint.CreateLocalPos0Attr().Set(fixed_joint_postion)
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(0.7071, 0.7071, 0, 0))
    # Local position1: Object(Body1) 기준으로 joint를 고정할 위치 / rotation
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(0.0, 0.0, 0.0, 0.0))

    # object의 collision mesh를 approximate하기 위한 방법 설정
    model_prim = my_world.scene.get_object(geometry_prim.name)
    # none, convexDecomposition, convexHull, boundingSphere, boundingCube, meshSimplification, sdf, sphereFill
    model_prim.set_collision_approximation('convexDecomposition')       
    

# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/OmniverseKit_Persp')

# GUI 상에서 보는 view point의 위치를 지정
eye = np.array([3.0, 3.0, 4.0])
target = np.array([0.5, 0.5, 1.0])
set_camera_view(eye, target, '/OmniverseKit_Persp', viewport)

# 생성한 world 에서 physics simulation step​
while simulation_app.is_running():
    # 생성한 world 에서 physics simulation step​
    my_world.step(render=True)

    if my_world.is_playing():
        # step이 0일때, world와 controller를 reset
        if my_world.current_time_step_index == 0:
            my_world.reset()
# 시뮬레이션 종료
simulation_app.close()