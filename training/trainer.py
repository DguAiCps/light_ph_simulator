"""
학습 루프 관리.

Trainer 클래스.
"""

from typing import Dict, List
from pathlib import Path
import numpy as np
import torch

from .buffer import ReplayBuffer
from .sac import SACAgent

try:
    from ..core import MultiAgentEnv
    from ..config import TrainingConfig
except ImportError:
    from core import MultiAgentEnv
    from config import TrainingConfig


class Trainer:
    """
    학습 루프 관리 클래스.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        agent: SACAgent,
        buffer: ReplayBuffer,
        config: TrainingConfig
    ):
        """
        Args:
            env: 환경
            agent: SAC 에이전트
            buffer: 리플레이 버퍼
            config: 학습 설정
        """
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.config = config

        self.total_steps = 0
        self.episode_count = 0

    def train(
        self,
        num_episodes: int,
        checkpoint_interval: int = 0,
        checkpoint_path: str = "checkpoints"
    ) -> Dict:
        """
        메인 학습 루프.

        Args:
            num_episodes: 학습 에피소드 수
            checkpoint_interval: 체크포인트 저장 주기 (0이면 저장 안함)
            checkpoint_path: 체크포인트 저장 경로

        Returns:
            학습 통계 (rewards, losses 등)
        """
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'critic_losses': [],
            'actor_losses': [],
            'success_rates': []
        }

        for episode in range(num_episodes):
            # 에피소드 실행
            episode_info = self._run_episode(explore=True)

            history['episode_rewards'].append(episode_info['total_reward'])
            history['episode_lengths'].append(episode_info['steps'])

            self.episode_count += 1

            # Warmup 이후 학습 (버퍼가 batch_size 이상일 때만)
            min_samples = max(self.config.warmup_steps, self.config.batch_size)

            # DEBUG
            if episode % 10 == 0:
                print(f"  [DEBUG] Buffer: {len(self.buffer)} / min_samples: {min_samples}")

            if len(self.buffer) >= min_samples:
                # 에피소드당 업데이트 (10스텝당 1회로 줄임)
                num_updates = max(1, episode_info['steps'] // 10)
                critic_losses = []
                actor_losses = []

                for update_idx in range(num_updates):
                    try:
                        batch = self.buffer.sample(self.config.batch_size)
                        losses = self.agent.update(batch, buffer=self.buffer)
                        critic_losses.append(losses['critic_loss'])
                        if losses['actor_loss'] != 0.0:  # Actor가 업데이트된 경우 (0이 아닐 때)
                            actor_losses.append(losses['actor_loss'])
                    except Exception as e:
                        print(f"  [ERROR] Update failed at {update_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        break

                history['critic_losses'].append(np.mean(critic_losses))
                history['actor_losses'].append(np.mean(actor_losses) if actor_losses else 0.0)

            # 로깅
            if episode % 10 == 0:
                avg_reward = np.mean(history['episode_rewards'][-10:])
                log_msg = (f"[Ep {episode}] Steps: {self.total_steps} | "
                           f"Reward: {episode_info['total_reward']:.2f} | "
                           f"Avg: {avg_reward:.2f}")

                # Loss 정보 추가 (학습 중일 때만)
                if history['critic_losses']:
                    critic_loss = history['critic_losses'][-1]
                    actor_loss = history['actor_losses'][-1]
                    log_msg += f" | C_loss: {critic_loss:.4f} | A_loss: {actor_loss:.4f}"

                log_msg += f" | Success: {episode_info['success']}"
                print(log_msg)

            # 주기적 체크포인트 저장
            if checkpoint_interval > 0 and (episode + 1) % checkpoint_interval == 0:
                ckpt_file = f"{checkpoint_path}/light_ph_ep{episode + 1}.pt"
                self.save_checkpoint(ckpt_file)

        return history

    def evaluate(self, num_episodes: int) -> Dict:
        """
        평가 (탐험 없이).

        Args:
            num_episodes: 평가 에피소드 수

        Returns:
            평가 통계 (success_rate, avg_reward 등)
        """
        self.agent.actor.eval()

        rewards = []
        successes = []
        lengths = []

        for _ in range(num_episodes):
            episode_info = self._run_episode(explore=False)
            rewards.append(episode_info['total_reward'])
            successes.append(episode_info['success'])
            lengths.append(episode_info['steps'])

        self.agent.actor.train()

        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': np.mean(successes),
            'avg_length': np.mean(lengths)
        }

    def _run_episode(self, explore: bool = True) -> Dict:
        """
        단일 에피소드 실행.

        Args:
            explore: 탐험 여부

        Returns:
            에피소드 정보 (total_reward, steps, success 등)
        """
        obs = self.env.reset()
        states = obs['states']
        goals = obs['goals']

        total_reward = np.zeros(self.env.env_config.num_agents)
        step = 0
        done = False

        while not done:
            # 각 에이전트 액션 계산
            actions = []
            for i in range(self.env.env_config.num_agents):
                state = states[i]
                goal = goals[i]
                neighbors = [states[j] for j in range(self.env.env_config.num_agents) if j != i]

                action = self.agent.select_action(state, goal, neighbors, explore=explore)
                actions.append(action)

            actions = np.array(actions)

            # 환경 스텝
            next_obs, rewards, done, info = self.env.step(actions)

            # 버퍼에 저장 (학습 중일 때만)
            if explore:
                self.buffer.push(
                    state={'states': states, 'goals': goals},
                    action=actions,
                    reward=rewards,
                    next_state={'states': next_obs['states'], 'goals': next_obs['goals']},
                    done=done
                )

            total_reward += rewards
            step += 1
            self.total_steps += 1

            states = next_obs['states']
            goals = next_obs['goals']

        success = all(info['arrived'])

        return {
            'total_reward': total_reward.mean(),
            'steps': step,
            'success': success,
            'arrived': info['arrived'],
            'collisions': len(info['collisions'])
        }

    def save_checkpoint(self, path: str):
        """
        체크포인트 저장.

        Args:
            path: 저장 경로
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.agent.save(path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        체크포인트 로드.

        Args:
            path: 로드 경로
        """
        self.agent.load(path)
        print(f"Checkpoint loaded from {path}")
