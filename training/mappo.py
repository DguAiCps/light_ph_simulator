"""
MAPPO (Multi-Agent PPO) 알고리즘.

CTDE 패러다임: Centralized Training, Decentralized Execution.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .buffer import RolloutBuffer

try:
    from ..networks import Actor, Critic
    from ..core import MultiAgentEnv
except ImportError:
    from networks import Actor, Critic
    from core import MultiAgentEnv


class MAPPO:
    """
    Multi-Agent PPO 알고리즘.

    각 에이전트가 공유 Actor를 사용 (Parameter Sharing).
    Critic은 global state를 사용 (Centralized).
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        actor_hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: str = 'cpu'
    ):
        """
        Args:
            env: 멀티에이전트 환경
            actor_hidden_dim: Actor hidden 차원
            critic_hidden_dim: Critic hidden 차원
            lr_actor: Actor 학습률
            lr_critic: Critic 학습률
            gamma: 할인율
            gae_lambda: GAE lambda
            clip_eps: PPO 클리핑 epsilon
            entropy_coef: Entropy 보너스 계수
            value_loss_coef: Value loss 계수
            max_grad_norm: Gradient clipping
            ppo_epochs: PPO 업데이트 에포크
            batch_size: 미니배치 크기
            buffer_size: 버퍼 크기
            device: 디바이스
        """
        self.env = env
        self.num_agents = env.env_config.num_agents
        self.device = device

        # Observation/action 차원 계산
        # obs: [자기상태(4) + 목표오프셋(2) + 이웃상태(4*(N-1))]
        self.obs_dim = 4 + 2 + 4 * (self.num_agents - 1)
        self.action_dim = 2

        # Global state: 모든 에이전트 상태 concat
        self.state_dim = self.obs_dim * self.num_agents

        # 네트워크 생성 (Parameter Sharing)
        self.actor = Actor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=actor_hidden_dim
        ).to(device)

        self.critic = Critic(
            state_dim=self.state_dim,
            hidden_dim=critic_hidden_dim
        ).to(device)

        # Optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 하이퍼파라미터
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # 버퍼
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device
        )

        # 통계
        self.total_steps = 0
        self.episode_count = 0

    def get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        환경에서 observation 추출.

        Returns:
            obs: 각 에이전트 observation (num_agents, obs_dim)
            state: Global state (state_dim,)
        """
        obs = np.array([
            self.env.get_observation(i) for i in range(self.num_agents)
        ])
        state = obs.flatten()
        return obs, state

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        모든 에이전트의 액션 선택.

        Args:
            obs: 각 에이전트 observation (num_agents, obs_dim)
            deterministic: 결정적 행동 여부

        Returns:
            actions: 선택된 액션 (num_agents, action_dim)
            log_probs: 로그 확률 (num_agents,)
            values: Value 추정 (num_agents,)
        """
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

            # Actor: 각 에이전트 액션 (parameter sharing이므로 batch로 처리)
            actions, log_probs, _ = self.actor.get_action(obs_tensor, deterministic)

            # Critic: global state로 value 추정
            state = obs.flatten()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            value = self.critic(state_tensor.unsqueeze(0))  # (1, 1)

            # 모든 에이전트 동일 value (centralized critic)
            values = np.full(self.num_agents, value.item())

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values
        )

    def collect_rollout(self) -> Dict[str, float]:
        """
        버퍼가 찰 때까지 데이터 수집.

        Returns:
            rollout 통계 (에피소드 보상 등)
        """
        self.buffer.reset()
        episode_rewards = []
        episode_lengths = []
        current_ep_reward = np.zeros(self.num_agents)
        current_ep_length = 0

        obs_dict = self.env.reset()
        obs, state = self.get_observations()

        while self.buffer.ptr < self.buffer.buffer_size:
            # 액션 선택
            actions, log_probs, values = self.select_action(obs)

            # 환경 스텝
            next_obs_dict, rewards, done, info = self.env.step(actions)

            # 버퍼에 저장
            self.buffer.store(
                obs=obs,
                state=state,
                action=actions,
                reward=rewards,
                done=done,
                log_prob=log_probs,
                value=values
            )

            current_ep_reward += rewards
            current_ep_length += 1
            self.total_steps += 1

            # 다음 상태
            obs, state = self.get_observations()

            if done:
                # GAE 계산
                self.buffer.finish_path(np.zeros(self.num_agents))

                episode_rewards.append(current_ep_reward.mean())
                episode_lengths.append(current_ep_length)
                self.episode_count += 1

                current_ep_reward = np.zeros(self.num_agents)
                current_ep_length = 0

                # 환경 리셋
                obs_dict = self.env.reset()
                obs, state = self.get_observations()

        # 마지막 경로 처리 (에피소드 중간에 버퍼가 찬 경우)
        if self.buffer.path_start_idx < self.buffer.ptr:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                last_value = self.critic(state_tensor.unsqueeze(0)).item()
            self.buffer.finish_path(np.full(self.num_agents, last_value))

        return {
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0,
            'num_episodes': len(episode_rewards)
        }

    def update(self) -> Dict[str, float]:
        """
        PPO 업데이트.

        Returns:
            학습 통계
        """
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get(self.batch_size):
                obs, states, actions, old_log_probs, advantages, returns = batch

                # Actor 업데이트
                new_log_probs, entropy = self.actor.evaluate_action(obs, actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                actor_total_loss = actor_loss + self.entropy_coef * entropy_loss

                self.actor_optimizer.zero_grad()
                actor_total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic 업데이트
                values = self.critic(states).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values, returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates
        }

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 10,
        save_interval: int = 100,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        학습 메인 루프.

        Args:
            total_timesteps: 총 학습 스텝
            log_interval: 로깅 간격 (iteration)
            save_interval: 저장 간격 (iteration)
            save_path: 모델 저장 경로

        Returns:
            학습 히스토리
        """
        history = []
        iteration = 0

        while self.total_steps < total_timesteps:
            # Rollout 수집
            rollout_stats = self.collect_rollout()

            # PPO 업데이트
            update_stats = self.update()

            # 통계 기록
            stats = {
                'iteration': iteration,
                'total_steps': self.total_steps,
                'episodes': self.episode_count,
                **rollout_stats,
                **update_stats
            }
            history.append(stats)

            # 로깅
            if iteration % log_interval == 0:
                print(f"[Iter {iteration}] Steps: {self.total_steps}, "
                      f"Episodes: {self.episode_count}, "
                      f"Reward: {rollout_stats['mean_reward']:.2f}, "
                      f"Actor Loss: {update_stats['actor_loss']:.4f}, "
                      f"Critic Loss: {update_stats['critic_loss']:.4f}")

            # 저장
            if save_path and iteration % save_interval == 0:
                self.save(f"{save_path}/mappo_iter_{iteration}.pt")

            iteration += 1

        return history

    def save(self, path: str):
        """모델 저장."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count
        }, path)

    def load(self, path: str):
        """모델 로드."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
