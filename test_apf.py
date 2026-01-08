"""
APF 컨트롤러로 4개 에이전트가 자동으로 목표로 이동하는 테스트.

사용법:
    python test_apf.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from config import RobotConfig, EnvConfig
from core import MultiAgentEnv
from controller import APFController
from utils.visualization import Visualizer


def main():
    # 설정 생성
    robot_config = RobotConfig()
    env_config = EnvConfig(num_agents=4)

    # 환경 및 시각화 생성
    env = MultiAgentEnv(env_config, robot_config)
    vis = Visualizer(env_config, robot_config)

    # APF 컨트롤러 생성 (각 에이전트마다)
    controllers = [
        APFController(k_att=2.0, k_rep=1.5, d0=0.5)
        for _ in range(env_config.num_agents)
    ]

    # 환경 초기화
    obs = env.reset()
    vis.reset()

    print("=== APF Controller Test ===")
    print(f"Agents: {env_config.num_agents}")
    print("Press Q to quit, R to reset")
    print("===========================")

    running = True

    def on_key(event):
        nonlocal running, obs
        if event.key == 'q':
            running = False
        elif event.key == 'r':
            obs = env.reset()
            vis.trajectories = [[] for _ in range(env_config.num_agents)]
            for ctrl in controllers:
                ctrl.reset()

    vis.fig.canvas.mpl_connect('key_press_event', on_key)

    # 메인 루프
    step = 0
    while running:
        states = obs['states']
        goals = obs['goals']

        # 각 에이전트의 액션 계산
        actions = []
        for i in range(env_config.num_agents):
            # 이웃 = 자기 제외한 다른 에이전트들
            neighbors = [states[j] for j in range(env_config.num_agents) if j != i]

            # APF로 액션 계산
            action = controllers[i].compute_action(states[i], goals[i], neighbors)
            actions.append(action)

        actions = np.array(actions)

        # 환경 스텝
        obs, rewards, done, info = env.step(actions)

        # 시각화
        vis.render(obs['states'], obs['goals'], show_trajectory=True)

        # 상태 표시
        arrived_count = sum(info['arrived'])
        collision_count = len(info['collisions'])
        vis.ax.set_title(
            f'Step: {step} | Arrived: {arrived_count}/{env_config.num_agents} | '
            f'Collisions: {collision_count}'
        )

        step += 1

        if done:
            if all(info['arrived']):
                print(f"All agents reached goals in {step} steps!")
            else:
                print(f"Episode ended at step {step}. Arrived: {arrived_count}/{env_config.num_agents}")

            # 잠시 대기 후 리셋
            plt.pause(1.0)
            obs = env.reset()
            vis.trajectories = [[] for _ in range(env_config.num_agents)]
            step = 0

        plt.pause(0.02)

    vis.close()
    print("Test ended.")


if __name__ == "__main__":
    main()
