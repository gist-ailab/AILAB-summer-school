import os
import numpy as np
import cv2

# Segmentation visualization
seg_labels = [0, 1, 2]
seg_colors = [np.array([0, 0, 0]), np.array([255, 0, 0]), np.array([0, 255, 0])]

def save_image(rgb, depth, seg, file_name):
    seg_rgb = np.zeros((480, 640, 3), dtype=np.uint8)    
    # post-processing
    # depth normalization
    min_depth, max_depth = depth.min(), depth.max()
    depth = (depth - min_depth) / (max_depth - min_depth) * 255
    depth = depth.astype('uint8')

    # segmentation visualization
    num_classes = 3
    for class_num in range(1, num_classes):
        seg_rgb[seg == class_num] = seg_colors[class_num]


    # rgb_image = Image.fromarray(rgb)
    cv2.imwrite(file_name + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(file_name + "_depth.png", cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
    cv2.imwrite(file_name + "_seg.png", cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR))

from omni.isaac.kit import SimulationApp
config = {
    'width': 640,
    'height': 480,
    'headless': False
}
simulation_app = SimulationApp(config)
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)
import carb
import omni.isaac.core.utils.carb as carb_utils
settings = carb.settings.get_settings()
carb_utils.set_carb_setting(settings, "/persistent/isaac/asset_root/default", "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1")

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.semantics import add_update_semantics


# World 선언
my_world = World(stage_units_in_meters=1.0)

# Initialize the World
my_world.scene.add_default_ground_plane()

cube = my_world.scene.add(
    DynamicCuboid(
        prim_path="/cube1", # should be unique
        name="cube1", # should be unique, used to access the object
        position=[0, 0, 1.0],
        scale=[0.6, 0.5, 0.2],
        size=1.0,
    )
)

my_camera = Camera(                                             
    prim_path="/World/Camera",
    frequency=20,                                               
    resolution=(640, 480),                                    
    position=[0.48176, 0.13541, 0.71],
    orientation=[0.5,-0.5,0.5,0.5]
)                                                               
my_camera.initialize()

my_camera.set_focal_length(1.93)
my_camera.set_focus_distance(4)                                 
my_camera.set_horizontal_aperture(2.65)                        
my_camera.set_vertical_aperture(1.48)     

my_camera.set_clipping_range(0.01, 10000)                       

my_camera.add_distance_to_camera_to_frame()
my_camera.add_instance_segmentation_to_frame()


# Initialize the World and Camera
my_world.reset()
is_semantic_initialized = False
while not is_semantic_initialized:
    my_world.step(render=True)
    if my_camera.get_current_frame()["instance_segmentation"] is not None:
        is_semantic_initialized = True
my_world.reset()

# Simulation Loop
total_episodes = 10
max_episode_steps = 100
save_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2.3images")
os.makedirs(save_root, exist_ok=True)

for i in range(10):
    # Reset the world
    my_world.reset()
    print(f"Episode: {i}")
    add_update_semantics(prim=cube.prim, semantic_label=f"{i}")
    
    for j in range(100):
        print(f"Step: {j}")
        file_name =  f"episode_{i}_step_{j}"
        rgb_image = my_camera.get_rgba()
    
        current_frame = my_camera.get_current_frame()

        distance_image = current_frame["distance_to_camera"]
        instance_segmentation_image = current_frame["instance_segmentation"]["data"]
        instance_segmentation_dict = current_frame["instance_segmentation"]["info"]["idToSemantics"]
        print(instance_segmentation_dict)

        save_image(rgb_image, distance_image, instance_segmentation_image, os.path.join(save_root, file_name))

        my_world.step(render=True)
    
    
# SimulationApp Close
simulation_app.close()
print("Simulation is Closed")

