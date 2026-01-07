"""
MAPPO Actor 네트워크.

연속 액션 공간용 Gaussian 정책.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


class Actor(nn.Module):
    """
    MAPPO Actor 네트워크.

    Observation을 받아 액션 분포(Gaussian)를 출력.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        """
        Args:
            obs_dim: Observation 차원
            action_dim: Action 차원 (2: Fx, Fy)
            hidden_dim: Hidden layer 크기
            num_layers: Hidden layer 개수
            log_std_min: Log std 최소값
            log_std_max: Log std 최대값
        """
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim

        self.feature = nn.Sequential(*layers)

        # Mean과 log_std 출력
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Orthogonal 초기화."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor (batch, obs_dim)

        Returns:
            mean: 액션 평균 (batch, action_dim)
            std: 액션 표준편차 (batch, action_dim)
        """
        features = self.feature(obs)

        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        return mean, std

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        액션 샘플링.

        Args:
            obs: Observation tensor
            deterministic: True면 mean 반환

        Returns:
            action: 샘플된 액션
            log_prob: 로그 확률
            entropy: 엔트로피
        """
        mean, std = self.forward(obs)

        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
            entropy = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(mean, std)
            action = dist.rsample()  # Reparameterization trick
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        주어진 액션의 log_prob과 entropy 계산.

        Args:
            obs: Observation tensor
            action: 평가할 액션

        Returns:
            log_prob: 로그 확률
            entropy: 엔트로피
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy
