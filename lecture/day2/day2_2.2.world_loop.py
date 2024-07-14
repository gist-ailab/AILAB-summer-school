from omni.isaac.kit import SimulationApp
config = {
    'width': 1920,
    'height': 1080,
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
cube1 = my_world.scene.get_object('cube1')
# check cube1 is same as cube
print(cube == cube1)



# Simulation Loop
time = 0
while simulation_app.is_running():
    print(f"Simulation Application is Running... {time}")
    # Reset the world
    my_world.reset()
    world_step = 0
    
    # World Loop
    for i in range(100):
        print(f"Wolrd Step: {world_step}")
        my_world.step(render=True)
        print(cube.get_angular_velocity())
        print(cube.get_world_pose())
        world_step += 1
    time += 1
    
# SimulationApp Close
simulation_app.close()
print("Simulation is Closed")

