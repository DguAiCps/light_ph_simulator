"""
Experience Replay Buffer 및 MAPPO용 Rollout Buffer.

ReplayBuffer: SAC용 off-policy 버퍼
RolloutBuffer: MAPPO용 on-policy 버퍼 (GAE 계산)
"""

import numpy as np
import torch
from typing import Generator, Tuple, Dict, Any
from collections import deque
import random


class RolloutBuffer:
    """
    On-policy 데이터 저장용 버퍼.

    에피소드 끝에 GAE 계산 후 학습에 사용.
    """

    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = 'cpu'
    ):
        """
        Args:
            buffer_size: 버퍼 크기 (스텝 수)
            num_agents: 에이전트 수
            obs_dim: 개별 observation 차원
            state_dim: Global state 차원 (critic용)
            action_dim: Action 차원
            gamma: 할인율
            gae_lambda: GAE lambda
            device: 디바이스
        """
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.ptr = 0
        self.path_start_idx = 0

        # 버퍼 할당
        self.obs = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_agents), dtype=np.float32)

        # GAE 계산 결과
        self.advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_agents), dtype=np.float32)

    def store(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: bool,
        log_prob: np.ndarray,
        value: np.ndarray
    ):
        """
        한 스텝 데이터 저장.

        Args:
            obs: 각 에이전트 observation (num_agents, obs_dim)
            state: Global state (state_dim,)
            action: 각 에이전트 액션 (num_agents, action_dim)
            reward: 각 에이전트 보상 (num_agents,)
            done: 에피소드 종료 여부
            log_prob: 각 에이전트 log_prob (num_agents,)
            value: 각 에이전트 value (num_agents,)
        """
        assert self.ptr < self.buffer_size

        self.obs[self.ptr] = obs
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value

        self.ptr += 1

    def finish_path(self, last_value: np.ndarray):
        """
        에피소드/경로 종료 시 GAE 계산.

        Args:
            last_value: 마지막 상태의 value (num_agents,)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]

        path_len = self.ptr - self.path_start_idx

        # GAE 계산
        advantages = np.zeros((path_len, self.num_agents), dtype=np.float32)
        last_gae = np.zeros(self.num_agents, dtype=np.float32)

        for t in reversed(range(path_len)):
            if t == path_len - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + values

        self.path_start_idx = self.ptr

    def get(
        self,
        batch_size: int = None
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        미니배치 생성기.

        Args:
            batch_size: 미니배치 크기 (None이면 전체)

        Yields:
            (obs, states, actions, log_probs, advantages, returns) 튜플
        """
        assert self.ptr == self.buffer_size, "Buffer not full"

        # Advantage 정규화
        advantages = self.advantages.copy()
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Returns 정규화 (Critic loss 안정화)
        returns = self.returns.copy()
        ret_mean = returns.mean()
        ret_std = returns.std() + 1e-8
        returns = (returns - ret_mean) / ret_std

        if batch_size is None:
            batch_size = self.buffer_size * self.num_agents

        # 데이터 flatten (buffer_size * num_agents, ...)
        total_size = self.buffer_size * self.num_agents
        obs_flat = self.obs.reshape(total_size, self.obs_dim)
        actions_flat = self.actions.reshape(total_size, self.action_dim)
        log_probs_flat = self.log_probs.reshape(total_size)
        advantages_flat = advantages.reshape(total_size)
        returns_flat = returns.reshape(total_size)  # 정규화된 returns 사용

        # States는 각 에이전트마다 같으므로 반복
        states_flat = np.repeat(self.states, self.num_agents, axis=0)

        # 셔플 인덱스
        indices = np.random.permutation(total_size)

        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_idx = indices[start:end]

            yield (
                torch.tensor(obs_flat[batch_idx], device=self.device),
                torch.tensor(states_flat[batch_idx], device=self.device),
                torch.tensor(actions_flat[batch_idx], device=self.device),
                torch.tensor(log_probs_flat[batch_idx], device=self.device),
                torch.tensor(advantages_flat[batch_idx], device=self.device),
                torch.tensor(returns_flat[batch_idx], device=self.device)
            )

    def reset(self):
        """버퍼 초기화."""
        self.ptr = 0
        self.path_start_idx = 0


class ReplayBuffer:
    """
    Experience Replay Buffer for SAC.

    Off-policy 학습용 경험 저장.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: 버퍼 최대 크기
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        reward: np.ndarray,
        next_state: Dict[str, Any],
        done: bool
    ):
        """
        경험 저장.

        Args:
            state: 현재 상태 (dict: states, goals)
            action: 수행한 액션 (N, 2)
            reward: 받은 보상 (N,)
            next_state: 다음 상태 (dict: states, goals)
            done: 종료 여부
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        랜덤 배치 샘플링.

        Args:
            batch_size: 샘플 수

        Returns:
            (states, actions, rewards, next_states, dones) 튜플
        """
        batch = random.sample(self.buffer, batch_size)

        states = [item[0] for item in batch]
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = [item[3] for item in batch]
        dones = np.array([item[4] for item in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """현재 버퍼 크기."""
        return len(self.buffer)
