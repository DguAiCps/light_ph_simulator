"""
Vectorized Environment용 Trainer.

여러 환경에서 동시에 데이터 수집하여 학습 속도 향상.
"""

from typing import Dict, List
from pathlib import Path
import numpy as np

from .buffer import ReplayBuffer
from .sac import SACAgent

try:
    from ..core import VectorizedEnv
    from ..config import TrainingConfig
except ImportError:
    from core import VectorizedEnv
    from config import TrainingConfig


class VectorizedTrainer:
    """
    Vectorized Environment를 사용하는 Trainer.
    """

    def __init__(
        self,
        vec_env: VectorizedEnv,
        agent: SACAgent,
        buffer: ReplayBuffer,
        config: TrainingConfig
    ):
        """
        Args:
            vec_env: Vectorized 환경
            agent: SAC 에이전트
            buffer: 리플레이 버퍼
            config: 학습 설정
        """
        self.vec_env = vec_env
        self.agent = agent
        self.buffer = buffer
        self.config = config

        self.total_steps = 0
        self.episode_count = 0
        self.num_envs = vec_env.num_envs

    def train(
        self,
        total_steps: int,
        checkpoint_interval: int = 0,
        checkpoint_path: str = "checkpoints"
    ) -> Dict:
        """
        메인 학습 루프 (스텝 기반).

        Args:
            total_steps: 총 학습 스텝 수
            checkpoint_interval: 체크포인트 저장 주기 (스텝, 0이면 저장 안함)
            checkpoint_path: 체크포인트 저장 경로

        Returns:
            학습 통계
        """
        history = {
            'episode_rewards': [],
            'critic_losses': [],
            'actor_losses': [],
        }

        # 환경 초기화
        obs_list = self.vec_env.reset()

        # 각 환경의 에피소드 보상 추적
        episode_rewards = [0.0] * self.num_envs

        min_samples = max(self.config.warmup_steps, self.config.batch_size)

        while self.total_steps < total_steps:
            # 모든 환경에서 액션 계산
            actions_list = []
            for env_idx in range(self.num_envs):
                obs = obs_list[env_idx]
                states = obs['states']
                goals = obs['goals']

                # 각 에이전트 액션
                actions = []
                for i in range(self.vec_env.get_num_agents()):
                    state = states[i]
                    goal = goals[i]
                    neighbors = [states[j] for j in range(self.vec_env.get_num_agents()) if j != i]
                    action = self.agent.select_action(state, goal, neighbors, explore=True)
                    actions.append(action)

                actions_list.append(np.array(actions))

            # 스텝 실행 및 데이터 수집
            transitions, next_obs_list, dones_list, infos_list = self.vec_env.step_and_collect(
                actions_list, obs_list
            )

            # 버퍼에 저장
            for trans in transitions:
                self.buffer.push(
                    state=trans['state'],
                    action=trans['action'],
                    reward=trans['reward'],
                    next_state=trans['next_state'],
                    done=trans['done']
                )

            # 보상 누적
            for env_idx in range(self.num_envs):
                episode_rewards[env_idx] += transitions[env_idx]['reward'].mean()

                if dones_list[env_idx]:
                    history['episode_rewards'].append(episode_rewards[env_idx])
                    episode_rewards[env_idx] = 0.0
                    self.episode_count += 1

            self.total_steps += self.num_envs
            obs_list = next_obs_list

            # 학습
            if len(self.buffer) >= min_samples:
                batch = self.buffer.sample(self.config.batch_size)
                losses = self.agent.update(batch, buffer=self.buffer)
                history['critic_losses'].append(losses['critic_loss'])
                if losses['actor_loss'] != 0.0:
                    history['actor_losses'].append(losses['actor_loss'])

            # 로깅 (1000 스텝마다)
            if self.total_steps % 1000 < self.num_envs:
                avg_reward = np.mean(history['episode_rewards'][-10:]) if history['episode_rewards'] else 0.0
                log_msg = (f"[Step {self.total_steps}] Episodes: {self.episode_count} | "
                           f"Avg Reward: {avg_reward:.2f}")

                if history['critic_losses']:
                    log_msg += f" | C_loss: {history['critic_losses'][-1]:.4f}"
                if history['actor_losses']:
                    log_msg += f" | A_loss: {history['actor_losses'][-1]:.4f}"

                log_msg += f" | Buffer: {len(self.buffer)}"
                print(log_msg)

            # 체크포인트
            if checkpoint_interval > 0 and self.total_steps % checkpoint_interval < self.num_envs:
                ckpt_file = f"{checkpoint_path}/light_ph_step{self.total_steps}.pt"
                self.save_checkpoint(ckpt_file)

        return history

    def save_checkpoint(self, path: str):
        """체크포인트 저장."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """체크포인트 로드."""
        self.agent.load(path)
        print(f"Checkpoint loaded from {path}")
