"""
WASD로 로봇을 수동 조작하는 테스트 스크립트.

사용법:
    python test_manual.py

조작:
    W: 위로
    A: 왼쪽
    S: 아래
    D: 오른쪽
    Q: 종료
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# 모듈 경로 추가
sys.path.insert(0, '.')

from config import RobotConfig, EnvConfig
from core import MultiAgentEnv, AgentState
from utils.visualization import Visualizer


class ManualController:
    """WASD 키보드 입력을 받아 힘으로 변환."""

    def __init__(self, force_magnitude: float = 2.0):
        self.force_magnitude = force_magnitude
        self.current_force = np.zeros(2)
        self.running = True

    def on_key_press(self, event):
        """키 입력 처리."""
        if event.key == 'w':
            self.current_force = np.array([0.0, self.force_magnitude])
        elif event.key == 's':
            self.current_force = np.array([0.0, -self.force_magnitude])
        elif event.key == 'a':
            self.current_force = np.array([-self.force_magnitude, 0.0])
        elif event.key == 'd':
            self.current_force = np.array([self.force_magnitude, 0.0])
        elif event.key == 'q':
            self.running = False

    def on_key_release(self, event):
        """키 뗌 처리."""
        if event.key in ['w', 'a', 's', 'd']:
            self.current_force = np.zeros(2)

    def get_force(self) -> np.ndarray:
        """현재 힘 반환."""
        return self.current_force.copy()


def main():
    # 설정 생성
    robot_config = RobotConfig()
    env_config = EnvConfig(num_agents=1)  # 에이전트 1개만

    # 환경 및 시각화 생성
    env = MultiAgentEnv(env_config, robot_config)
    vis = Visualizer(env_config, robot_config)

    # 환경 초기화
    obs = env.reset()
    vis.reset()

    # 컨트롤러 생성
    controller = ManualController(force_magnitude=3.0)

    # 키보드 이벤트 연결
    vis.fig.canvas.mpl_connect('key_press_event', controller.on_key_press)
    vis.fig.canvas.mpl_connect('key_release_event', controller.on_key_release)

    print("=== Manual Control Test ===")
    print("W: Up, A: Left, S: Down, D: Right")
    print("Q: Quit")
    print("===========================")

    # 메인 루프
    while controller.running:
        # 현재 힘 가져오기
        force = controller.get_force()
        actions = np.array([force])  # shape (1, 2)

        # 벽 반발력 계산 (디버그용)
        wall_force = env._compute_wall_force(env.states[0])

        # 환경 스텝
        obs, rewards, done, info = env.step(actions)

        # 시각화
        vis.render(obs['states'], obs['goals'], show_trajectory=True)

        # 상태 출력
        state = obs['states'][0]
        goal = obs['goals'][0]
        dist = np.linalg.norm(state.position - goal)

        # 벽 반발력이 있으면 표시
        wall_str = ""
        if np.linalg.norm(wall_force) > 0.01:
            wall_str = f' | WallF: ({wall_force[0]:.1f}, {wall_force[1]:.1f})'

        vis.ax.set_title(
            f'Pos: ({state.position[0]:.2f}, {state.position[1]:.2f}) | '
            f'Vel: ({state.velocity[0]:.2f}, {state.velocity[1]:.2f}) | '
            f'Dist: {dist:.2f}{wall_str}'
        )

        if done:
            print("Goal reached!" if info['arrived'][0] else "Episode ended.")
            obs = env.reset()
            vis.trajectories = [[] for _ in range(env_config.num_agents)]

        plt.pause(0.02)

    vis.close()
    print("Test ended.")


if __name__ == "__main__":
    main()
