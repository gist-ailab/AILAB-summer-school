# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import typing

import numpy as np
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper


class PickPlace_Controller(BaseController):
    """
    A simple pick and place state machine for tutorials

    Each phase runs for 1 second, which is the internal time of the state machine

    Dt of each phase/ event step is defined

    - Phase 0: Move end_effector above the pre_picking xy position without orientation.
    - Phase 1: Move end_effector above the pre_picking_position at the 'end_effector_initial_height'.
    - Phase 2: Lower end_effector down to encircle the target object
    - Phase 3: Wait for Robot's inertia to settle.
    - Phase 4: close grip.
    - Phase 5: Move end_effector up again, keeping the grip tight (lifting the block).
    - Phase 6: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 7: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 8: loosen the grip.
    - Phase 9: Move end_effector vertically up again at the 'end_effector_initial_height'
    - Phase 10: Move end_effector towards the old xy position.

    Args:
        name (str): Name id of the controller
        cspace_controller (BaseController): a cartesian space controller that returns an ArticulationAction type
        gripper (Gripper): a gripper controller for open/ close actions.
        end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from (more info in phases above). If not defined, set to 0.3 meters. Defaults to None.
        events_dt (typing.Optional[typing.List[float]], optional): Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

    Raises:
        Exception: events dt need to be list or numpy array
        Exception: events dt need have length of 10
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.3 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [0.08, 0.008, 0.002, 0.1, 0.1, 0.0025, 0.001, 0.0025, 1, 0.008, 0.08]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 11:
                raise Exception("events dt length must be less than 10")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        return

    def is_paused(self) -> bool:
        """

        Returns:
            bool: True if the state machine is paused. Otherwise False.
        """
        return self._pause

    def get_current_event(self) -> int:
        """

        Returns:
            int: Current event/ phase of the state machine
        """
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        pre_picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        gripper_width: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray):  The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._event == 3:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 4:
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 8:
            target_joint_positions = self._gripper.forward(action="open")

        else:
            if self._event == 0:
                self._current_target_x = pre_picking_position[0]
                self._current_target_y = pre_picking_position[1]
                self._h0 = pre_picking_position[2] + 0.3
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
                
            elif self._event == 1:
                # opening the gripper to gripper width
                if gripper_width is not None:
                    # when closing the gripper, positions are np.array([np.pi*2/9, -np.pi*2/9])
                    # we are using robotiq-140f gripper, full opend gripper width is 0.14 meter
                    # so we are converting the gripper width to the joint positions
                    if gripper_width > 0.14:
                        gripper_width = 0.14
                    gripper_width = np.array([(gripper_width / 0.14) * (np.pi * 2 / 9), (-gripper_width / 0.14) * (np.pi * 2 / 9)])
                    self._gripper.set_joint_positions(gripper_width)
                    
                self._current_target_x = pre_picking_position[0]
                self._current_target_y = pre_picking_position[1]
                self._h0 = pre_picking_position[2]
                
            elif self._event == 2:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]
                
            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
            )
            target_height = self._get_target_hs(placing_position[2])
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
            )
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            if self._event == 5:
                self._event +=1
            self._t = 0
        return target_joint_positions

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        if self._event < 6:
            return 0
        elif self._event == 6:
            return self._mix_sin(self._t)
        elif self._event in [7, 8, 9]:
            return 1.0
        elif self._event == 10:
            return 1
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h0
        elif self._event == 1:
            h = self._h0
        elif self._event == 2:
            h = self._h0
        elif self._event == 4:
            h = self._h0
        elif self._event == 5:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 6:
            h = self._h1
        elif self._event == 7:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 8:
            h = target_height
        elif self._event == 9:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 10:
            h = self._h1
        else:
            raise ValueError()
        return h

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from. If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 11:
                raise Exception("events dt length must be less than 10")
        return

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        """Pauses the state machine's time and phase."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase."""
        self._pause = False
        return