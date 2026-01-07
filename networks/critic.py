"""
MAPPO Critic 네트워크.

Centralized Value Function (CTDE 패러다임).
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    MAPPO Critic 네트워크.

    Global state를 받아 Value 출력.
    CTDE: 훈련 시 모든 에이전트 정보 사용 가능.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        """
        Args:
            state_dim: Global state 차원 (모든 에이전트 정보 포함)
            hidden_dim: Hidden layer 크기
            num_layers: Hidden layer 개수
        """
        super().__init__()

        # Feature extractor
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim

        self.feature = nn.Sequential(*layers)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Orthogonal 초기화."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Global state tensor (batch, state_dim)

        Returns:
            value: State value (batch, 1)
        """
        features = self.feature(state)
        value = self.value_head(features)
        return value


class CriticWithAttention(nn.Module):
    """
    Attention 기반 Critic.

    각 에이전트의 observation을 개별 처리 후 attention으로 집계.
    더 scalable한 구조.
    """

    def __init__(
        self,
        agent_obs_dim: int,
        num_agents: int,
        hidden_dim: int = 64,
        num_heads: int = 4
    ):
        """
        Args:
            agent_obs_dim: 단일 에이전트 observation 차원
            num_agents: 에이전트 수
            hidden_dim: Hidden layer 크기
            num_heads: Attention head 수
        """
        super().__init__()

        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        # 각 에이전트 observation 인코딩
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        agent_obs: torch.Tensor,
        agent_idx: int = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            agent_obs: 모든 에이전트 observation (batch, num_agents, agent_obs_dim)
            agent_idx: 특정 에이전트 관점의 value 계산 (None이면 평균)

        Returns:
            value: State value (batch, 1) or (batch, num_agents, 1)
        """
        batch_size = agent_obs.shape[0]

        # 각 에이전트 인코딩
        encoded = self.agent_encoder(agent_obs)  # (batch, num_agents, hidden_dim)

        # Self-attention
        attended, _ = self.attention(encoded, encoded, encoded)

        if agent_idx is not None:
            # 특정 에이전트의 value
            agent_repr = attended[:, agent_idx, :]  # (batch, hidden_dim)
            value = self.value_head(agent_repr)
        else:
            # 모든 에이전트의 value
            values = self.value_head(attended)  # (batch, num_agents, 1)
            value = values.mean(dim=1)  # (batch, 1)

        return value
