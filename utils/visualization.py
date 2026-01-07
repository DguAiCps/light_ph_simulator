"""
Matplotlib 기반 시각화.

에이전트 이동 및 환경 렌더링.
"""

from typing import List, Optional, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

if TYPE_CHECKING:
    from ..config import EnvConfig, RobotConfig
    from ..core.dynamics import AgentState


# 에이전트 색상
AGENT_COLORS = list(mcolors.TABLEAU_COLORS.values())[:10]


class Visualizer:
    """Matplotlib 기반 시각화 클래스."""

    def __init__(self, env_config: 'EnvConfig', robot_config: 'RobotConfig'):
        """
        Args:
            env_config: 환경 설정
            robot_config: 로봇 설정
        """
        self.env_config = env_config
        self.robot_config = robot_config

        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

        # 궤적 저장
        self.trajectories: List[List[np.ndarray]] = []

        # 애니메이션용 프레임 저장
        self.frames: List[dict] = []

    def reset(self) -> None:
        """시각화 초기화."""
        if self.fig is not None:
            plt.close(self.fig)

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.trajectories = [[] for _ in range(self.env_config.num_agents)]
        self.frames = []

        self._setup_axes()

    def _setup_axes(self) -> None:
        """축 설정."""
        self.ax.set_xlim(-0.2, self.env_config.width + 0.2)
        self.ax.set_ylim(-0.2, self.env_config.height + 0.2)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Light-pH Simulator')
        self.ax.grid(True, alpha=0.3)

    def render(
        self,
        states: List['AgentState'],
        goals: np.ndarray,
        show_trajectory: bool = True,
        save_frame: bool = False
    ) -> None:
        """
        현재 상태 렌더링.

        Args:
            states: 에이전트 상태들
            goals: 목표 위치들
            show_trajectory: 궤적 표시 여부
            save_frame: 애니메이션용 프레임 저장 여부
        """
        if self.fig is None:
            self.reset()

        self.ax.clear()
        self._setup_axes()

        # 벽 그리기 (사각형 경계)
        wall = Rectangle(
            (0, 0),
            self.env_config.width,
            self.env_config.height,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        self.ax.add_patch(wall)

        # 궤적 업데이트 및 그리기
        for i, state in enumerate(states):
            self.trajectories[i].append(state.position.copy())

            if show_trajectory and len(self.trajectories[i]) > 1:
                traj = np.array(self.trajectories[i])
                self.ax.plot(
                    traj[:, 0], traj[:, 1],
                    color=AGENT_COLORS[i % len(AGENT_COLORS)],
                    alpha=0.5,
                    linewidth=1
                )

        # 목표 그리기 (X 마커)
        for i, goal in enumerate(goals):
            self.ax.plot(
                goal[0], goal[1],
                'x',
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                markersize=15,
                markeredgewidth=3,
                label=f'Goal {i}'
            )

        # 에이전트 그리기 (점 + 원)
        for i, state in enumerate(states):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]

            # 로봇 몸체 (원)
            circle = Circle(
                state.position,
                self.robot_config.radius,
                fill=True,
                facecolor=color,
                edgecolor='black',
                alpha=0.7,
                linewidth=1.5
            )
            self.ax.add_patch(circle)

            # 중심점
            self.ax.plot(
                state.position[0], state.position[1],
                'o',
                color='white',
                markersize=4
            )

            # 속도 방향 화살표
            if np.linalg.norm(state.velocity) > 0.01:
                vel_normalized = state.velocity / np.linalg.norm(state.velocity)
                self.ax.arrow(
                    state.position[0],
                    state.position[1],
                    vel_normalized[0] * self.robot_config.radius * 1.5,
                    vel_normalized[1] * self.robot_config.radius * 1.5,
                    head_width=0.05,
                    head_length=0.03,
                    fc='white',
                    ec='white'
                )

            # 에이전트 번호
            self.ax.text(
                state.position[0],
                state.position[1] + self.robot_config.radius + 0.1,
                f'{i}',
                ha='center',
                fontsize=10,
                fontweight='bold'
            )

        # 프레임 저장
        if save_frame:
            self.frames.append({
                'states': [s.copy() for s in states],
                'goals': goals.copy()
            })

        plt.pause(0.001)

    def save_animation(self, path: str, fps: int = 20) -> None:
        """
        에피소드 애니메이션 저장.

        Args:
            path: 저장 경로 (.gif 또는 .mp4)
            fps: 프레임 레이트
        """
        if not self.frames:
            print("No frames to save. Call render() with save_frame=True first.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        def init():
            ax.set_xlim(-0.2, self.env_config.width + 0.2)
            ax.set_ylim(-0.2, self.env_config.height + 0.2)
            ax.set_aspect('equal')
            return []

        def update(frame_idx):
            ax.clear()
            ax.set_xlim(-0.2, self.env_config.width + 0.2)
            ax.set_ylim(-0.2, self.env_config.height + 0.2)
            ax.set_aspect('equal')
            ax.set_title(f'Step {frame_idx}')
            ax.grid(True, alpha=0.3)

            frame = self.frames[frame_idx]
            states = frame['states']
            goals = frame['goals']

            # 벽
            wall = Rectangle(
                (0, 0),
                self.env_config.width,
                self.env_config.height,
                fill=False,
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(wall)

            # 궤적 (현재 프레임까지)
            for i in range(len(states)):
                traj = [self.frames[j]['states'][i].position for j in range(frame_idx + 1)]
                if len(traj) > 1:
                    traj = np.array(traj)
                    ax.plot(
                        traj[:, 0], traj[:, 1],
                        color=AGENT_COLORS[i % len(AGENT_COLORS)],
                        alpha=0.5,
                        linewidth=1
                    )

            # 목표
            for i, goal in enumerate(goals):
                ax.plot(
                    goal[0], goal[1],
                    'x',
                    color=AGENT_COLORS[i % len(AGENT_COLORS)],
                    markersize=15,
                    markeredgewidth=3
                )

            # 에이전트
            for i, state in enumerate(states):
                color = AGENT_COLORS[i % len(AGENT_COLORS)]
                circle = Circle(
                    state.position,
                    self.robot_config.radius,
                    fill=True,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.7
                )
                ax.add_patch(circle)

            return []

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=len(self.frames),
            interval=1000 // fps,
            blit=False
        )

        if path.endswith('.gif'):
            anim.save(path, writer='pillow', fps=fps)
        else:
            anim.save(path, writer='ffmpeg', fps=fps)

        plt.close(fig)
        print(f"Animation saved to {path}")

    def close(self) -> None:
        """Figure 닫기."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
