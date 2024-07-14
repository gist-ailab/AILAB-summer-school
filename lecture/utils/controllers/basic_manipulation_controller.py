# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.universal_robots.controllers import RMPFlowController


class BasicManipulationController(BaseController):
    """ 
        A simple end-effector position, open/close control state machine for tutorials

        The phase runs for 1 second, which is the internal time of the state machine

        Dt of each phase/ event step is defined

        - Phase 0: Move end_effector at the target position.

        Args:
            name (str): Name id of the controller
            cspace_controller (BaseController): a cartesian space controller that returns an ArticulationAction type
            gripper (Gripper): a gripper controller for open/ close actions.
                from (more info in phases above). If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional): Dt of each phase/ event step. 1 phase dt has to
                be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 1
        """

    def __init__(
        self,
        name: str,              # 컨트롤러 이름
        gripper: Gripper,       # 그리퍼
        cspace_controller: BaseController = RMPFlowController,  # 모션 컨트롤러
        events_dt: typing.Optional[typing.List[float]] = None,  # 이벤트 dt
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0     # event 설정
        self._t = 0         # 타임 설정
        self._events_dt = events_dt # 이벤트 dt 설정. dt가 작을 수록 한 phase안에 더 많은 모션을 수행
        if self._events_dt is None:
            self._events_dt = [0.008]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 1:
                raise Exception("events dt length must be less than 1")
        self._cspace_controller = cspace_controller     # 모션 컨트롤러 설정
        self._gripper = gripper                         # 그리퍼 설정
        self._pause = False                             # 시뮬레이션 멈춤 해제
        return

    # end effector를 원하는 위치로 가게 하는 함수
    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            target_position (np.ndarray):  The end-effector's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0.14])
            
        # 시뮬레이션이 멈춰 있거나 끝났을 때 실행
        if self._pause or self.is_done():
            # 시뮬레이션 멈춤
            self.pause()
            # Action에 None 리턴
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        
        # end effector offset을 고려한 타겟 x,y,z 위치 계산
        # end effector offset은 end effector에서 그리퍼 끝까지의 길이때문에 발생하는 offset을 의미
        position_target = np.array(
            [
                target_position[0] + end_effector_offset[0],
                target_position[1] + end_effector_offset[1],
                target_position[2] + end_effector_offset[2],
            ]
        )
        
        # end effector orientation 설정
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        # ArticulationAction 생성
        target_joint_positions = self._cspace_controller.forward(
            target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
        )
        
        # 이벤트 시간이 _event_dt만큼 흐르게 함
        # _event_dt가 쌓여서 단위시간 1 만큼 흘렀다면 phase 종료 및 이벤트 시간 초기화
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        
        # end effector가 어떤 위치(x,y,z)와 orientation을 향해 
        # 어떤 속도로 가야 하는지에 대한 정보 리턴
        return target_joint_positions
    
    # 그리퍼의 손가락을 여는 함수
    def open(
        self,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0.14])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        # 그리퍼 open 액션을 통해 타겟 joint position 계산
        target_joint_positions = self._gripper.forward(action="open")
        
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
            
        # 그리퍼의 타겟 joint position 리턴
        return target_joint_positions

    # 그리퍼의 손가락을 닫는 함수
    def close(
        self,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0.14])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        # 그리퍼 close 액션을 통해 타겟 joint position 계산
        target_joint_positions = self._gripper.forward(action="close")
        
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
            
        # 그리퍼의 타겟 joint position 리턴
        return target_joint_positions


    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. 1 phase dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 1
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) or not isinstance(self._events_dt, list):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 1:
                raise Exception("events dt length must be less than 1")
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
        """Pauses the state machine's time and phase.
        """
        self._pause = True
        return


    def resume(self) -> None:
        """Resumes the state machine's time and phase.
        """
        self._pause = False
        return
