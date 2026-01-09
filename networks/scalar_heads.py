"""
스칼라 파라미터 출력 헤드들.

NodeHead: 노드 레벨 스칼라 (k_g, d)
EdgeHead: 에이전트-에이전트 엣지 스칼라 (k, d, c)
ObstacleEdgeHead: 에이전트-장애물 엣지 스칼라 (k_obs)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


def bounded_activation(x: Tensor, low: float, high: float) -> Tensor:
    """
    Atan 기반 bounded 출력 활성화.

    Sigmoid 대비 장점: 극값에서도 gradient가 유지됨
    - sigmoid gradient at x=10: ~0.00005
    - atan gradient at x=10: ~0.01 (200배 큼)

    Args:
        x: 입력
        low: 하한
        high: 상한

    Returns:
        [low, high] 범위의 출력
    """
    # atan(x)/π + 0.5 → [0, 1] 범위로 매핑
    import math
    normalized = torch.atan(x) / math.pi + 0.5
    return low + (high - low) * normalized


class NodeHead(nn.Module):
    """
    노드 레벨 스칼라 출력: k_g, d

    k_g: 목표 인력 크기 [0.1, 15.0]
    d: 감쇠 계수 [0.1, 5.0]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        """
        Args:
            input_dim: 입력 차원 (hidden_dim from GAT)
            hidden_dim: 히든 차원
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # k_g 출력
        self.k_g_head = nn.Linear(hidden_dim, 1)
        # d 출력
        self.d_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            node_embedding: 노드 임베딩 (N, input_dim)

        Returns:
            k_g: 목표 스프링 상수 (N,), [0.1, 10.0]
            d: 감쇠 계수 (N,), [0.1, 5.0]
        """
        features = self.mlp(node_embedding)

        k_g = bounded_activation(self.k_g_head(features).squeeze(-1), 0.1, 15.0)
        d = bounded_activation(self.d_head(features).squeeze(-1), 0.1, 5.0)

        return k_g, d


class EdgeHead(nn.Module):
    """
    에이전트-에이전트 엣지 스칼라 출력: k, d, c

    k: 에이전트 간 스프링 상수 [0.01, 4.5] (k/r 공식에서 충돌 회피용 강한 척력 가능)
    d: 에이전트 간 감쇠 계수 [0.1, 5.0]
    c: 커플링 계수 [-2.0, 2.0]
    """

    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int = 32
    ):
        """
        Args:
            input_dim: 노드 embedding 차원
            edge_dim: 엣지 feature 차원
            hidden_dim: 히든 차원
        """
        super().__init__()

        # 입력: src_emb + dst_emb + edge_attr
        total_dim = input_dim * 2 + edge_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # k 출력
        self.k_head = nn.Linear(hidden_dim, 1)
        # d 출력
        self.d_head = nn.Linear(hidden_dim, 1)
        # c 출력
        self.c_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        src_emb: Tensor,
        dst_emb: Tensor,
        edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            src_emb: 소스 노드 embedding (E, input_dim)
            dst_emb: 타겟 노드 embedding (E, input_dim)
            edge_attr: 엣지 features (E, edge_dim)

        Returns:
            k: 스프링 상수 (E,), [0.01, 4.5]
            d: 감쇠 계수 (E,), [0.1, 5.0]
            c: 커플링 계수 (E,), [-2.0, 2.0]
        """
        # 결합
        combined = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        features = self.mlp(combined)

        k = bounded_activation(self.k_head(features).squeeze(-1), 0.01, 4.5)
        d = bounded_activation(self.d_head(features).squeeze(-1), 0.1, 5.0)
        c = bounded_activation(self.c_head(features).squeeze(-1), -2.0, 2.0)

        return k, d, c


class ObstacleEdgeHead(nn.Module):
    """
    에이전트-장애물 엣지 스칼라 출력: k_obs

    현재 시뮬레이터에서는 장애물 없음. 추후 확장용.

    k_obs: 장애물 반발 스프링 상수 [0.01, 3.0]
    """

    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int = 32
    ):
        """
        Args:
            input_dim: 노드 embedding 차원
            edge_dim: 엣지 feature 차원
            hidden_dim: 히든 차원
        """
        super().__init__()

        # 입력: src_emb (에이전트) + dst_emb (장애물) + edge_attr
        total_dim = input_dim * 2 + edge_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # k_obs 출력
        self.k_obs_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        src_emb: Tensor,
        dst_emb: Tensor,
        edge_attr: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Args:
            src_emb: 에이전트 embedding (E, input_dim)
            dst_emb: 장애물 embedding (E, input_dim)
            edge_attr: 엣지 features (E, edge_dim)

        Returns:
            k_obs: 장애물 반발 스프링 상수 (E,), [0.01, 0.5]
        """
        combined = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        features = self.mlp(combined)

        k_obs = bounded_activation(self.k_obs_head(features).squeeze(-1), 0.01, 3.0)

        return k_obs
