"""
Soft Actor-Critic 구현.

CTDE (Centralized Training, Decentralized Execution) 패러다임.
"""

import copy
from typing import List, Tuple, Dict, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..core.dynamics import AgentState

try:
    from ..controller import LightPHController
    from ..config import TrainingConfig
except ImportError:
    from controller import LightPHController
    from config import TrainingConfig


class CentralizedCritic(nn.Module):
    """
    CTDE용 중앙집중 Critic.

    모든 에이전트의 상태와 액션을 입력으로 받아 Q value 출력.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_agents: int
    ):
        """
        Args:
            state_dim: 단일 에이전트 상태 차원
            action_dim: 단일 에이전트 액션 차원
            hidden_dim: 히든 차원
            num_agents: 에이전트 수
        """
        super().__init__()

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 입력: pooled state + pooled action
        input_dim = state_dim + action_dim

        # Twin Q networks
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            states: 모든 에이전트 상태 (B, N, state_dim)
            actions: 모든 에이전트 액션 (B, N, action_dim)

        Returns:
            q1, q2: Q values (B, 1)
        """
        # Mean pooling over agents (논문 eq.16)
        state_pooled = states.mean(dim=1)  # (B, state_dim)
        action_pooled = actions.mean(dim=1)  # (B, action_dim)

        # 결합
        x = torch.cat([state_pooled, action_pooled], dim=-1)

        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2

    def q1_forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Q1만 forward."""
        state_pooled = states.mean(dim=1)
        action_pooled = actions.mean(dim=1)
        x = torch.cat([state_pooled, action_pooled], dim=-1)
        return self.q1(x)


class SACAgent:
    """
    Soft Actor-Critic Agent.

    Light-pH Controller를 Actor로 사용.
    """

    def __init__(
        self,
        actor: LightPHController,
        critic: CentralizedCritic,
        config: TrainingConfig,
        device: str = 'cuda'
    ):
        """
        Args:
            actor: Light-pH Controller (Actor)
            critic: Centralized Critic
            config: 학습 설정
            device: 디바이스
        """
        self.actor = actor
        self.critic = critic
        self.config = config
        self.device = device

        # Target network
        self.critic_target = copy.deepcopy(critic)
        self.critic_target.to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.get_parameters(),
            lr=config.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.lr_critic
        )

        # Entropy temperature (auto-tuning)
        self.target_entropy = -2.0  # -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr_actor)

        # Reward normalization - 스케일링 제거
        self.grad_clip = 1.0  # Gradient clipping
        self.q_clip = 1000.0  # Q-value clipping 범위 (넓게)

        # Actor 업데이트 설정
        self.actor_update_freq = 10  # Critic 10번당 Actor 1번
        self.actor_batch_size = 512  # Actor용 배치 (5090이면 더 키워도 됨)
        self.update_count = 0

        # 탐험 노이즈 감소 설정
        self.noise_start = 0.3  # 초기 노이즈
        self.noise_end = 0.05  # 최종 노이즈
        self.noise_decay_steps = 100000000  # 감소에 걸리는 스텝 수
        self.total_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        """Entropy coefficient."""
        return self.log_alpha.exp()

    def select_action(
        self,
        state: 'AgentState',
        goal: np.ndarray,
        neighbors: List['AgentState'],
        explore: bool = True
    ) -> np.ndarray:
        """
        액션 선택.

        Args:
            state: 현재 상태
            goal: 목표
            neighbors: 이웃들
            explore: 탐험 노이즈 추가 여부

        Returns:
            액션 [Fx, Fy]
        """
        # Light-pH 컨트롤러로 기본 액션 계산
        action = self.actor.compute_action(state, goal, neighbors)

        if explore:
            # 탐험 노이즈 (시간에 따라 감소)
            self.total_steps += 1
            decay_ratio = min(1.0, self.total_steps / self.noise_decay_steps)
            current_noise = self.noise_start + (self.noise_end - self.noise_start) * decay_ratio
            noise = np.random.normal(0, current_noise, size=action.shape)
            action = action + noise

        return action

    def update(self, batch: Tuple, buffer=None) -> Dict[str, float]:
        """
        네트워크 업데이트.

        Args:
            batch: 샘플된 배치 (states, actions, rewards, next_states, dones)
            buffer: 리플레이 버퍼 (actor 업데이트용 작은 배치 샘플링)

        Returns:
            {"critic_loss": float, "actor_loss": float, "alpha_loss": float}
        """
        critic_loss = self._update_critic(batch)

        # Actor는 덜 자주 업데이트 (느린 forward_differentiable 때문)
        self.update_count += 1
        actor_loss = 0.0
        alpha_loss = 0.0

        if self.update_count % self.actor_update_freq == 0:
            # Actor용 작은 배치 사용
            if buffer is not None and len(buffer) >= self.actor_batch_size:
                actor_batch = buffer.sample(self.actor_batch_size)
                actor_loss, alpha_loss = self._update_actor(actor_batch)
            else:
                # 배치 일부만 사용
                small_batch = tuple(x[:self.actor_batch_size] if hasattr(x, '__getitem__') else x for x in batch)
                actor_loss, alpha_loss = self._update_actor(small_batch)

        self._soft_update()

        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss
        }

    def _update_critic(self, batch: Tuple) -> float:
        """
        Critic 업데이트.

        Args:
            batch: (states, actions, rewards, next_states, dones)

        Returns:
            critic loss
        """
        states, actions, rewards, next_states, dones = batch

        # 텐서 변환
        states_tensor = self._states_to_tensor(states)
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = self._states_to_tensor(next_states)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # rewards: (B, N) -> (B, 1) mean (스케일링 제거)
        rewards_mean = rewards_tensor.mean(dim=1, keepdim=True)
        dones_expanded = dones_tensor.unsqueeze(1)

        with torch.no_grad():
            # 다음 상태에서 현재 policy로 액션 계산 (정석 SAC)
            num_agents = self.critic.num_agents
            next_actions = self.actor.forward_differentiable(next_states, num_agents)

            # Target Q value
            q1_target, q2_target = self.critic_target(next_states_tensor, next_actions)
            q_target = torch.min(q1_target, q2_target)

            # TD target (스케일링 없이)
            target = rewards_mean + self.config.gamma * (1 - dones_expanded) * q_target

        # Current Q values
        q1, q2 = self.critic(states_tensor, actions_tensor)

        # Critic loss
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, batch: Tuple) -> Tuple[float, float]:
        """
        Actor 업데이트.

        Args:
            batch: (states, actions, rewards, next_states, dones)

        Returns:
            (actor_loss, alpha_loss)
        """
        states, actions, _, _, _ = batch

        # Differentiable forward pass로 새로운 액션 계산
        # 이렇게 해야 gradient가 actor로 전파됨
        num_agents = self.critic.num_agents
        new_actions = self.actor.forward_differentiable(states, num_agents)

        # states도 텐서로 변환
        states_tensor = self._states_to_tensor(states)

        # Actor loss: -Q(s, π(s))
        # 새로 계산한 액션 사용 (gradient 연결됨)
        q1 = self.critic.q1_forward(states_tensor, new_actions)
        actor_loss = -q1.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping (actor는 여러 모듈로 구성)
        for module in [self.actor.gat_backbone, self.actor.node_head, self.actor.edge_head]:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # Alpha loss (entropy temperature)
        # 단순화: 고정 alpha 사용
        alpha_loss = 0.0

        return actor_loss.item(), alpha_loss

    def _soft_update(self):
        """Target network soft update."""
        tau = self.config.tau
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def _states_to_tensor(self, states: List[Dict]) -> torch.Tensor:
        """
        상태 딕셔너리 리스트를 텐서로 변환.

        Args:
            states: 상태 딕셔너리 리스트

        Returns:
            (B, N, state_dim) 텐서
        """
        batch = []
        for state_dict in states:
            agent_states = []
            for agent_state in state_dict['states']:
                # [x, y, vx, vy]
                s = np.concatenate([agent_state.position, agent_state.velocity])
                agent_states.append(s)
            batch.append(agent_states)

        return torch.tensor(np.array(batch), dtype=torch.float32, device=self.device)

    def save(self, path: str):
        """모델 저장."""
        torch.save({
            'actor_gat': self.actor.gat_backbone.state_dict(),
            'actor_node_head': self.actor.node_head.state_dict(),
            'actor_edge_head': self.actor.edge_head.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)

    def load(self, path: str):
        """모델 로드."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.gat_backbone.load_state_dict(checkpoint['actor_gat'])
        self.actor.node_head.load_state_dict(checkpoint['actor_node_head'])
        self.actor.edge_head.load_state_dict(checkpoint['actor_edge_head'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
