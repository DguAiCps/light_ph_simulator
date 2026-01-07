"""
물리 시뮬레이션 관련 클래스.

2차 적분기 + 감쇠 물리 모델 구현.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..config import RobotConfig


@dataclass
class AgentState:
    """단일 에이전트의 상태를 표현."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 위치 [x, y]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 속도 [vx, vy]

    def __post_init__(self):
        """numpy 배열로 변환 보장."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)

    def copy(self) -> 'AgentState':
        """상태 복사본 생성."""
        return AgentState(
            position=self.position.copy(),
            velocity=self.velocity.copy()
        )


class Dynamics:
    """
    2차 적분기 + 감쇠 물리 모델.

    운동 방정식:
        q̇ = v
        v̇ = (1/m)F - βv

    여기서:
        q: 위치
        v: 속도
        m: 질량
        F: 입력 힘
        β: 감쇠 계수 (damping / mass)
    """

    def __init__(self, robot_config: 'RobotConfig'):
        """
        Args:
            robot_config: 로봇 물리 파라미터
        """
        self.mass = robot_config.mass
        self.damping = robot_config.damping
        self.radius = robot_config.radius
        self.max_velocity = robot_config.max_velocity
        self.max_acceleration = robot_config.max_acceleration

        # 감쇠 비율 (β = damping / mass)
        self.beta = self.damping / self.mass

    def step(self, state: AgentState, force: np.ndarray, dt: float) -> AgentState:
        """
        한 timestep 물리 시뮬레이션 진행.

        Args:
            state: 현재 상태
            force: 입력 힘 [Fx, Fy]
            dt: timestep

        Returns:
            다음 상태
        """
        force = np.asarray(force, dtype=np.float64)

        # 가속도 계산: a = F/m - β*v
        acceleration = force / self.mass - self.beta * state.velocity

        # 가속도 클리핑
        acc_magnitude = np.linalg.norm(acceleration)
        if acc_magnitude > self.max_acceleration:
            acceleration = acceleration * (self.max_acceleration / acc_magnitude)

        # 속도 업데이트: v_new = v + a * dt
        new_velocity = state.velocity + acceleration * dt

        # 속도 클리핑
        vel_magnitude = np.linalg.norm(new_velocity)
        if vel_magnitude > self.max_velocity:
            new_velocity = new_velocity * (self.max_velocity / vel_magnitude)

        # 위치 업데이트: q_new = q + v * dt (반 암시적 오일러)
        new_position = state.position + new_velocity * dt

        return AgentState(position=new_position, velocity=new_velocity)

    def compute_kinetic_energy(self, state: AgentState) -> float:
        """
        운동 에너지 계산.

        Args:
            state: 에이전트 상태

        Returns:
            운동 에너지 K = (1/2) * m * |v|^2
        """
        speed_squared = np.dot(state.velocity, state.velocity)
        return 0.5 * self.mass * speed_squared
