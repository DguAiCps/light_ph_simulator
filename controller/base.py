"""
컨트롤러 인터페이스 정의.

모든 컨트롤러가 상속해야 하는 추상 클래스.
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.dynamics import AgentState


class BaseController(ABC):
    """모든 컨트롤러가 상속해야 하는 추상 클래스."""

    @abstractmethod
    def compute_action(
        self,
        state: 'AgentState',
        goal: np.ndarray,
        neighbors: List['AgentState']
    ) -> np.ndarray:
        """
        제어 입력 계산.

        Args:
            state: 현재 에이전트 상태
            goal: 목표 위치 [x, y]
            neighbors: 이웃 에이전트 상태들

        Returns:
            힘 입력 [Fx, Fy]
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """에피소드 시작 시 컨트롤러 상태 초기화."""
        pass
