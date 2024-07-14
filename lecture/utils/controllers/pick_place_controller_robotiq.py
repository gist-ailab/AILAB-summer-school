from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import Articulation
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
# import omni.isaac.manipulators.controllers as manipulators_controllers
from utils.controllers.RMPFflow_pickplace import RMPFlowController
from day3.pickplace_controller import PickPlace_Controller
import numpy as np
from typing import Optional, List


class PickPlaceController(PickPlace_Controller):
    """[summary]

        Args:
            name (str): [description]
            surface_gripper (SurfaceGripper): [description]
            robot_articulation(Articulation): [description]
            events_dt (Optional[List[float]], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: Articulation,
        events_dt: Optional[List[float]] = None,
        end_effector_initial_height: Optional[float] = None,
    ) -> None:
        if events_dt is None:
            events_dt = [0.01, 0.01, 0.01, 1, 0.01, 0.1, 0.05, 0.005, 1, 0.01, 0.1]
        PickPlace_Controller.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation, attach_gripper=True
            ),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=end_effector_initial_height,
        )
        return

    def forward(
        self,
        picking_position: np.ndarray,
        pre_picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: Optional[np.ndarray] = None,
        end_effector_orientation: Optional[np.ndarray] = None,
        gripper_width: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """[summary]

        Args:
            picking_position (np.ndarray): [description]
            placing_position (np.ndarray): [description]
            current_joint_positions (np.ndarray): [description]
            end_effector_offset (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.

        Returns:
            ArticulationAction: [description]
        """
        if end_effector_orientation is None:
           end_effector_orientation = euler_angles_to_quat(np.array([np.pi, 0, np.pi]))
        
        # if end_effector_offset is None:
        #     end_effector_offset = np.array([0, 0, 0.23])


        return super().forward(
            picking_position,
            pre_picking_position,
            placing_position,
            current_joint_positions,
            end_effector_offset=end_effector_offset,
            end_effector_orientation=end_effector_orientation,
            gripper_width=gripper_width,
        )
