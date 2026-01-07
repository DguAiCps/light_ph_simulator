"""
Vectorized Environment.

여러 환경을 동시에 실행하여 데이터 수집 속도 향상.
"""

from typing import List, Tuple, Dict, Callable
import numpy as np

from .environment import MultiAgentEnv

try:
    from ..config import EnvConfig, RobotConfig
except ImportError:
    from config import EnvConfig, RobotConfig


class VectorizedEnv:
    """
    여러 환경을 동시에 실행하는 래퍼.

    DummyVecEnv 스타일 - 같은 프로세스에서 순차 실행하지만 API 통일.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        robot_config: RobotConfig,
        num_envs: int = 8
    ):
        """
        Args:
            env_config: 환경 설정
            robot_config: 로봇 설정
            num_envs: 병렬 환경 수
        """
        self.num_envs = num_envs
        self.env_config = env_config
        self.robot_config = robot_config

        # 환경들 생성
        self.envs = [
            MultiAgentEnv(env_config, robot_config)
            for _ in range(num_envs)
        ]

        # 각 환경의 done 상태 추적
        self.dones = [False] * num_envs

    def reset(self) -> List[Dict]:
        """
        모든 환경 리셋.

        Returns:
            obs_list: 각 환경의 observation 리스트
        """
        obs_list = []
        for i, env in enumerate(self.envs):
            obs = env.reset()
            obs_list.append(obs)
            self.dones[i] = False
        return obs_list

    def step(
        self,
        actions_list: List[np.ndarray]
    ) -> Tuple[List[Dict], List[np.ndarray], List[bool], List[Dict]]:
        """
        모든 환경에서 스텝 실행.

        Args:
            actions_list: 각 환경의 액션 리스트 [(num_agents, 2), ...]

        Returns:
            obs_list: 각 환경의 다음 observation
            rewards_list: 각 환경의 reward
            dones_list: 각 환경의 done 여부
            infos_list: 각 환경의 info
        """
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []

        for i, (env, actions) in enumerate(zip(self.envs, actions_list)):
            if self.dones[i]:
                # 이미 종료된 환경은 자동 리셋
                obs = env.reset()
                obs_list.append(obs)
                rewards_list.append(np.zeros(self.env_config.num_agents))
                dones_list.append(False)
                infos_list.append({'arrived': [False] * self.env_config.num_agents, 'collisions': []})
                self.dones[i] = False
            else:
                obs, rewards, done, info = env.step(actions)
                obs_list.append(obs)
                rewards_list.append(rewards)
                dones_list.append(done)
                infos_list.append(info)
                self.dones[i] = done

        return obs_list, rewards_list, dones_list, infos_list

    def step_and_collect(
        self,
        actions_list: List[np.ndarray],
        current_obs_list: List[Dict]
    ) -> List[Dict]:
        """
        스텝 실행 + 버퍼용 transition 데이터 수집.

        Args:
            actions_list: 각 환경의 액션
            current_obs_list: 현재 observation (step 전)

        Returns:
            transitions: 버퍼에 저장할 transition 리스트
        """
        next_obs_list, rewards_list, dones_list, infos_list = self.step(actions_list)

        transitions = []
        for i in range(self.num_envs):
            transition = {
                'state': {
                    'states': current_obs_list[i]['states'],
                    'goals': current_obs_list[i]['goals']
                },
                'action': actions_list[i],
                'reward': rewards_list[i],
                'next_state': {
                    'states': next_obs_list[i]['states'],
                    'goals': next_obs_list[i]['goals']
                },
                'done': dones_list[i]
            }
            transitions.append(transition)

        return transitions, next_obs_list, dones_list, infos_list

    def get_num_agents(self) -> int:
        """에이전트 수 반환."""
        return self.env_config.num_agents
