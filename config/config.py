"""
Light-pH Simulator 설정 모듈.

설정 관련 데이터 클래스 및 유틸리티 함수.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class RobotConfig:
    """로봇의 물리적 파라미터를 정의하는 데이터 클래스."""

    mass: float = 1.0  # 로봇 질량 (kg)
    damping: float = 5.0  # 감쇠 계수 (높을수록 관성 적음)
    radius: float = 0.105  # 로봇 반지름 (m), TurtleBot3 크기 반영
    max_velocity: float = 0.22  # 최대 속도 (m/s)
    max_acceleration: float = 1.0  # 최대 가속도 (m/s²)


@dataclass
class EnvConfig:
    """환경 설정 데이터 클래스."""

    width: float = 1.5  # 환경 너비 (m)
    height: float = 1.5  # 환경 높이 (m)
    num_agents: int = 4  # 에이전트 수
    dt: float = 0.05  # 시뮬레이션 timestep (s)
    max_steps: int = 300  # 에피소드 최대 스텝
    goal_threshold: float = 0.15  # 도착 판정 거리 (m)
    collision_penalty: float = -10.0  # 충돌 시 페널티
    goal_reward: float = 10.0  # 도착 시 보상
    wall_stiffness: float = 50.0  # 벽 강성 k_wall (낮을수록 부드러움)
    wall_damping: float = 20.0  # 벽 감쇠 b_wall (높을수록 덜 튕김)


@dataclass
class GATConfig:
    """GAT 네트워크 설정 데이터 클래스."""

    node_input_dim: int = 12  # 노드 feature 차원
    edge_input_dim: int = 7  # 엣지 feature 차원
    hidden_dim: int = 64  # 히든 레이어 차원
    num_heads: int = 4  # Attention head 수
    num_layers: int = 3  # GAT 레이어 수
    dropout: float = 0.0  # Dropout 비율


@dataclass
class TrainingConfig:
    """학습 관련 설정 데이터 클래스."""

    lr_actor: float = 3e-4  # Actor 학습률
    lr_critic: float = 3e-4  # Critic 학습률
    gamma: float = 0.99  # 할인율
    tau: float = 0.005  # Soft update 계수
    batch_size: int = 256  # 배치 크기
    buffer_size: int = 100000  # Replay buffer 크기
    warmup_steps: int = 1000  # 학습 시작 전 스텝


@dataclass
class Config:
    """전체 설정을 담는 데이터 클래스."""

    robot: RobotConfig = field(default_factory=RobotConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str) -> Config:
    """
    YAML 파일에서 설정을 로드.

    Args:
        path: 설정 파일 경로

    Returns:
        Config 객체
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    config = Config()

    if data is None:
        return config

    if 'robot' in data:
        config.robot = RobotConfig(**data['robot'])
    if 'env' in data:
        config.env = EnvConfig(**data['env'])
    if 'gat' in data:
        config.gat = GATConfig(**data['gat'])
    if 'training' in data:
        config.training = TrainingConfig(**data['training'])

    return config


def save_config(config: Config, path: str) -> None:
    """
    설정을 YAML 파일로 저장.

    Args:
        config: Config 객체
        path: 저장 경로
    """
    data = {
        'robot': asdict(config.robot),
        'env': asdict(config.env),
        'gat': asdict(config.gat),
        'training': asdict(config.training),
    }

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
