"""
그래프 생성 유틸리티.

torch_geometric용 그래프 데이터 생성.
"""

from typing import List, TYPE_CHECKING
import numpy as np
import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    from ..core.dynamics import AgentState


def compute_edge_features(src_state: 'AgentState', dst_state: 'AgentState') -> np.ndarray:
    """
    단일 엣지의 feature 계산.

    Args:
        src_state: 소스 에이전트 상태
        dst_state: 타겟 에이전트 상태

    Returns:
        엣지 features (7,):
            - Δq: 상대 위치 (2)
            - Δv: 상대 속도 (2)
            - r: 거리 (1)
            - q̂: 단위 방향 벡터 (2)
    """
    # 상대 위치
    delta_q = dst_state.position - src_state.position

    # 상대 속도
    delta_v = dst_state.velocity - src_state.velocity

    # 거리
    r = np.linalg.norm(delta_q)

    # 단위 방향 벡터 (0으로 나누기 방지)
    if r > 1e-6:
        q_hat = delta_q / r
    else:
        q_hat = np.zeros(2)

    return np.concatenate([delta_q, delta_v, [r], q_hat])


def build_agent_graph(
    states: List['AgentState'],
    goals: np.ndarray,
    agent_idx: int = None
) -> Data:
    """
    Fully connected 에이전트 그래프 생성.

    Args:
        states: 모든 에이전트 상태
        goals: 목표 위치들 (N, 2)
        agent_idx: 특정 에이전트 관점 (None이면 전체 그래프)

    Returns:
        torch_geometric.data.Data:
            - x: 노드 features (N, 12)
            - edge_index: 엣지 연결 (2, E)
            - edge_attr: 엣지 features (E, 7)
    """
    num_agents = len(states)

    # Node features 생성
    # [type(3), state(4), mission(3), goal_offset(2)] = 12
    node_features = []

    for i, state in enumerate(states):
        # type: one-hot [is_self, is_neighbor, is_obstacle]
        if agent_idx is not None:
            type_vec = [1.0, 0.0, 0.0] if i == agent_idx else [0.0, 1.0, 0.0]
        else:
            type_vec = [1.0, 0.0, 0.0]  # 모두 에이전트

        # state: [x, y, vx, vy]
        state_vec = np.concatenate([state.position, state.velocity])

        # mission: [arrived, in_progress, not_started] - 간단히 in_progress로
        mission_vec = [0.0, 1.0, 0.0]

        # goal_offset: [dx, dy]
        goal_offset = goals[i] - state.position

        node_feature = np.concatenate([type_vec, state_vec, mission_vec, goal_offset])
        node_features.append(node_feature)

    node_features = np.array(node_features, dtype=np.float32)

    # Edge index 생성 (fully connected, self-loop 제외)
    edge_src = []
    edge_dst = []
    edge_features = []

    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                edge_src.append(i)
                edge_dst.append(j)
                edge_features.append(compute_edge_features(states[i], states[j]))

    edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    edge_features = np.array(edge_features, dtype=np.float32)

    # torch_geometric Data 객체 생성
    data = Data(
        x=torch.tensor(node_features),
        edge_index=torch.tensor(edge_index),
        edge_attr=torch.tensor(edge_features)
    )

    return data


def build_batch_graph(
    batch_states: List[List['AgentState']],
    batch_goals: List[np.ndarray]
) -> Data:
    """
    배치 그래프 생성 (학습용).

    Args:
        batch_states: 배치 내 모든 에피소드의 에이전트 상태들
        batch_goals: 배치 내 모든 에피소드의 목표 위치들

    Returns:
        배치된 torch_geometric.data.Data
    """
    from torch_geometric.data import Batch

    graphs = []
    for states, goals in zip(batch_states, batch_goals):
        graph = build_agent_graph(states, goals)
        graphs.append(graph)

    return Batch.from_data_list(graphs)
