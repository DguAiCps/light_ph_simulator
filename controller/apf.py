"""
Classical Artificial Potential Field 컨트롤러.

Baseline으로 사용.
"""

from typing import List, TYPE_CHECKING
import numpy as np

from .base import BaseController

if TYPE_CHECKING:
    from ..core.dynamics import AgentState


class APFController(BaseController):
    """
    Artificial Potential Field 기반 컨트롤러.

    목표를 향한 인력 + 이웃으로부터의 척력으로 제어 입력 계산.
    """

    def __init__(
        self,
        k_att: float = 1.0,
        k_rep: float = 1.0,
        d0: float = 0.5
    ):
        """
        Args:
            k_att: 인력 게인 (기본값: 1.0)
            k_rep: 척력 게인 (기본값: 1.0)
            d0: 척력 영향 거리 (기본값: 0.5)
        """
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0

    def compute_action(
        self,
        state: 'AgentState',
        goal: np.ndarray,
        neighbors: List['AgentState']
    ) -> np.ndarray:
        """
        APF 기반 제어 입력 계산.

        Args:
            state: 현재 상태
            goal: 목표 위치
            neighbors: 이웃들

        Returns:
            힘 [Fx, Fy]
        """
        # 인력
        f_att = self._attractive_force(state.position, goal)

        # 척력
        f_rep = self._repulsive_force(state.position, neighbors)

        return f_att + f_rep

    def _attractive_force(
        self,
        position: np.ndarray,
        goal: np.ndarray
    ) -> np.ndarray:
        """
        목표를 향한 인력 계산.

        F_att = k_att * (goal - position)

        Args:
            position: 현재 위치
            goal: 목표 위치

        Returns:
            인력 벡터
        """
        return self.k_att * (goal - position)

    def _repulsive_force(
        self,
        position: np.ndarray,
        neighbors: List['AgentState']
    ) -> np.ndarray:
        """
        이웃으로부터의 척력 계산.

        거리 < d0인 이웃에 대해 척력 합산.
        F_rep = k_rep * (1/d^2) * n_hat

        Args:
            position: 현재 위치
            neighbors: 이웃들

        Returns:
            총 척력 벡터
        """
        total_force = np.zeros(2)

        for neighbor in neighbors:
            diff = position - neighbor.position
            dist = np.linalg.norm(diff)

            # 영향 거리 내에 있는 경우만
            if 0 < dist < self.d0:
                # 단위 방향 벡터 (이웃에서 자기 방향)
                n_hat = diff / dist

                # 척력: 거리가 가까울수록 강함
                force_magnitude = self.k_rep * (1.0 / dist - 1.0 / self.d0) * (1.0 / dist**2)
                total_force += force_magnitude * n_hat

        return total_force

    def reset(self) -> None:
        """APF는 상태가 없으므로 pass."""
        pass
