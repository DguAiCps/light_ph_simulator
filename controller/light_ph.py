"""
Light-pH 컨트롤러.

GAT + Port-Hamiltonian Energy Shaping.
논문의 핵심 구현.
"""

from typing import List, Iterator, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from .base import BaseController

if TYPE_CHECKING:
    from ..core.dynamics import AgentState

try:
    from ..config import RobotConfig, GATConfig
    from ..networks import GATBackbone, NodeHead, EdgeHead
except ImportError:
    from config import RobotConfig, GATConfig
    from networks import GATBackbone, NodeHead, EdgeHead


class LightPHController(BaseController):
    """
    Light-pH 컨트롤러.

    BaseController를 상속. 논문의 핵심 구현.
    GAT로 이웃 정보를 처리하고 Port-Hamiltonian 제어 법칙 적용.
    """

    def __init__(
        self,
        gat_backbone: GATBackbone,
        node_head: NodeHead,
        edge_head: EdgeHead,
        robot_config: RobotConfig,
        device: str = 'cuda'
    ):
        """
        Args:
            gat_backbone: GAT 네트워크
            node_head: 노드 스칼라 출력 헤드
            edge_head: 엣지 스칼라 출력 헤드
            robot_config: 로봇 파라미터
            device: 디바이스
        """
        self.gat_backbone = gat_backbone
        self.node_head = node_head
        self.edge_head = edge_head
        self.robot_config = robot_config
        self.device = device

        # 모델을 디바이스로 이동
        self.gat_backbone.to(device)
        self.node_head.to(device)
        self.edge_head.to(device)

    def compute_action(
        self,
        state: 'AgentState',
        goal: np.ndarray,
        neighbors: List['AgentState']
    ) -> np.ndarray:
        """
        Light-pH 제어 입력 계산.

        Args:
            state: 현재 상태
            goal: 목표 위치
            neighbors: 이웃들

        Returns:
            힘 [Fx, Fy]
        """
        # 1. 그래프 구성
        graph_data = self._build_graph(state, goal, neighbors)
        graph_data = graph_data.to(self.device)

        # 2. GAT forward → node embeddings
        with torch.no_grad():
            node_embeddings = self.gat_backbone(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr
            )

            # 3. Scalar heads → k, d, c 값들
            # 자기 노드 (인덱스 0)의 k_g, d
            k_g, d_self = self.node_head(node_embeddings)
            k_g_self = k_g[0].item()
            d_self_val = d_self[0].item()

            # 엣지별 k, d, c (자기 노드에서 나가는 엣지들)
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr

            # 소스가 자기 자신(0)인 엣지들
            src_mask = edge_index[0] == 0
            if src_mask.sum() > 0:
                src_indices = edge_index[0][src_mask]
                dst_indices = edge_index[1][src_mask]

                src_emb = node_embeddings[src_indices]
                dst_emb = node_embeddings[dst_indices]
                edge_feat = edge_attr[src_mask]

                k_ij, d_ij, c_ij = self.edge_head(src_emb, dst_emb, edge_feat)
                k_ij = k_ij.cpu().numpy()
                d_ij = d_ij.cpu().numpy()
                c_ij = c_ij.cpu().numpy()
            else:
                k_ij = np.array([])
                d_ij = np.array([])
                c_ij = np.array([])

        # 4. Analytic ∇H 계산
        grad_H = self._compute_hamiltonian_gradient(
            state, goal, neighbors, k_ij, k_g_self
        )

        # 5. Port-Hamiltonian control law: u = (J - R) ∇H
        # 평균 d, c 사용 (또는 첫 번째 이웃)
        d_avg = d_ij.mean() if len(d_ij) > 0 else d_self_val
        c_avg = c_ij.mean() if len(c_ij) > 0 else 0.0

        J, R = self._assemble_pH_matrices(d_avg, c_avg)

        # u = (J - R) @ grad_H
        control_matrix = J - R
        u = control_matrix @ grad_H

        # 힘 추출 (grad_H = [∂H/∂q_x, ∂H/∂q_y, ∂H/∂v_x, ∂H/∂v_y])
        # 제어 입력은 속도에 대한 부분
        force = u[2:4]

        return force

    def _build_graph(
        self,
        state: 'AgentState',
        goal: np.ndarray,
        neighbors: List['AgentState']
    ) -> Data:
        """
        torch_geometric용 그래프 데이터 생성.

        Args:
            state: 자기 상태
            goal: 목표 위치
            neighbors: 이웃들

        Returns:
            torch_geometric.data.Data: 그래프 데이터
        """
        num_nodes = 1 + len(neighbors)  # 자기 + 이웃들

        # Node features: [type(3), state(4), mission(3), goal_offset(2)] = 12
        node_features = []

        # 자기 노드
        self_type = np.array([1, 0, 0])  # one-hot: [self, neighbor, obstacle]
        self_state = np.concatenate([state.position, state.velocity])  # 4
        self_mission = np.array([1, 0, 0])  # one-hot: [navigating, arrived, collision]
        goal_offset = goal - state.position  # 2
        self_node = np.concatenate([self_type, self_state, self_mission, goal_offset])
        node_features.append(self_node)

        # 이웃 노드들
        for neighbor in neighbors:
            neighbor_type = np.array([0, 1, 0])  # neighbor
            neighbor_state = np.concatenate([neighbor.position, neighbor.velocity])
            neighbor_mission = np.array([1, 0, 0])  # 모든 이웃도 navigating 가정
            # 이웃의 goal_offset은 알 수 없으므로 0
            neighbor_goal_offset = np.array([0, 0])
            neighbor_node = np.concatenate([
                neighbor_type, neighbor_state, neighbor_mission, neighbor_goal_offset
            ])
            node_features.append(neighbor_node)

        x = torch.tensor(np.array(node_features), dtype=torch.float32)

        # Edge features: [Δq(2), Δv(2), r(1), q̂(2)] = 7
        # Fully connected (self-loop 제외)
        edge_index = []
        edge_attr = []

        all_states = [state] + neighbors

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])

                    # 엣지 특성 계산
                    src_state = all_states[i]
                    dst_state = all_states[j]

                    delta_q = dst_state.position - src_state.position  # 2
                    delta_v = dst_state.velocity - src_state.velocity  # 2
                    r = np.linalg.norm(delta_q)  # 1
                    q_hat = delta_q / (r + 1e-6)  # 2 (단위 방향 벡터)

                    edge_feat = np.concatenate([delta_q, delta_v, [r], q_hat])
                    edge_attr.append(edge_feat)

        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # (2, E)
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float32)  # (E, 7)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _compute_hamiltonian_gradient(
        self,
        state: 'AgentState',
        goal: np.ndarray,
        neighbors: List['AgentState'],
        k: np.ndarray,
        k_g: float
    ) -> np.ndarray:
        """
        Analytic Hamiltonian gradient 계산 (논문 eq.9).

        Args:
            state: 현재 상태
            goal: 목표 위치
            neighbors: 이웃들
            k: 에이전트 간 스프링 상수 (len = num_neighbors)
            k_g: 목표 스프링 상수

        Returns:
            [∂H/∂q_x, ∂H/∂q_y, ∂H/∂v_x, ∂H/∂v_y]
        """
        q = state.position
        v = state.velocity
        m = self.robot_config.mass

        # ∂H/∂q = k_g(q - g) - Σ_j k_ij(q_i - q_j)
        # 목표: 인력 (q - goal 방향으로 끌림)
        # 이웃: 척력 (q - neighbor 반대 방향으로 밀림)
        dH_dq = k_g * (q - goal)

        for j, neighbor in enumerate(neighbors):
            if j < len(k):
                k_ij = k[j]
            else:
                k_ij = 1.0  # 기본값
            # 척력: 마이너스로 변경 (이웃에서 멀어지는 방향)
            dH_dq -= k_ij * (q - neighbor.position)

        # ∂H/∂v = mv
        dH_dv = m * v

        return np.concatenate([dH_dq, dH_dv])

    def _assemble_pH_matrices(
        self,
        d: float,
        c: float
    ) -> tuple:
        """
        Port-Hamiltonian J, R 행렬 조립.

        Args:
            d: 감쇠 계수
            c: 커플링 계수

        Returns:
            (J, R) 행렬 (4x4)
        """
        # J: skew-symmetric (에너지 보존)
        # 표준 Port-Hamiltonian 형태: J = [[0, I], [-I, 0]]
        J = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, 0, 0, c],
            [0, -1, -c, 0]
        ], dtype=np.float64)

        # R: positive semi-definite (에너지 소산)
        # R = [[0, 0], [0, d*I]]
        R = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, d, 0],
            [0, 0, 0, d]
        ], dtype=np.float64)

        return J, R

    def forward_differentiable(
        self,
        states_batch: List[dict],
        num_agents: int
    ) -> torch.Tensor:
        """
        Fully vectorized differentiable forward pass.

        Args:
            states_batch: 배치된 상태 딕셔너리 리스트
            num_agents: 에이전트 수

        Returns:
            actions: (B, N, 2) 텐서
        """
        batch_size = len(states_batch)
        total_graphs = batch_size * num_agents
        device = self.device
        num_neighbors = num_agents - 1

        # ========== 1. 모든 데이터 수집 및 텐서화 ==========
        all_graphs = []
        all_q = []
        all_v = []
        all_g = []
        all_neighbor_q = []

        for state_dict in states_batch:
            agent_states = state_dict['states']
            goals = state_dict['goals']

            for i in range(num_agents):
                state = agent_states[i]
                goal = goals[i]
                neighbors = [agent_states[j] for j in range(num_agents) if j != i]

                graph_data = self._build_graph(state, goal, neighbors)
                all_graphs.append(graph_data)

                all_q.append(state.position)
                all_v.append(state.velocity)
                all_g.append(goal)
                all_neighbor_q.append([n.position for n in neighbors])

        # 텐서 변환
        q_tensor = torch.tensor(np.array(all_q), dtype=torch.float32, device=device)
        v_tensor = torch.tensor(np.array(all_v), dtype=torch.float32, device=device)
        g_tensor = torch.tensor(np.array(all_g), dtype=torch.float32, device=device)
        neighbor_q_tensor = torch.tensor(np.array(all_neighbor_q), dtype=torch.float32, device=device)
        m = self.robot_config.mass

        # ========== 2. PyG Batch 및 GAT ==========
        batched_graph = Batch.from_data_list(all_graphs).to(device)

        all_node_embeddings = self.gat_backbone(
            batched_graph.x,
            batched_graph.edge_index,
            batched_graph.edge_attr
        )

        # ========== 3. Node head ==========
        all_k_g, all_d = self.node_head(all_node_embeddings)

        ego_indices = batched_graph.ptr[:-1]
        k_g_ego = all_k_g[ego_indices].view(-1, 1)

        # ========== 4. Edge head (Ego->Neighbor만) ==========
        src, dst = batched_graph.edge_index

        is_ego = torch.zeros(all_node_embeddings.size(0), dtype=torch.bool, device=device)
        is_ego[ego_indices] = True
        ego_edge_mask = is_ego[src]

        ego_src = src[ego_edge_mask]
        ego_dst = dst[ego_edge_mask]
        ego_edge_attr = batched_graph.edge_attr[ego_edge_mask]

        k_ij_all, d_ij_all, c_ij_all = self.edge_head(
            all_node_embeddings[ego_src],
            all_node_embeddings[ego_dst],
            ego_edge_attr
        )

        # ========== 5. Vectorized Interaction Force ==========
        k_ij_matrix = k_ij_all.view(total_graphs, num_neighbors)
        q_diff = q_tensor.unsqueeze(1) - neighbor_q_tensor
        weighted_diff = k_ij_matrix.unsqueeze(-1) * q_diff
        interaction_sum = weighted_diff.sum(dim=1)

        # ========== 6. Hamiltonian Gradient ==========
        dH_dq = k_g_ego * (q_tensor - g_tensor) - interaction_sum
        dH_dv = m * v_tensor

        # ========== 7. Vectorized d_avg, c_avg ==========
        d_ij_matrix = d_ij_all.view(total_graphs, num_neighbors)
        c_ij_matrix = c_ij_all.view(total_graphs, num_neighbors)
        d_avg = d_ij_matrix.mean(dim=1)
        c_avg = c_ij_matrix.mean(dim=1)

        # ========== 8. Control Law ==========
        dH_dq_x, dH_dq_y = dH_dq[:, 0], dH_dq[:, 1]
        dH_dv_x, dH_dv_y = dH_dv[:, 0], dH_dv[:, 1]

        u_2 = -dH_dq_x + c_avg * dH_dv_y - d_avg * dH_dv_x
        u_3 = -dH_dq_y - c_avg * dH_dv_x - d_avg * dH_dv_y

        all_actions = torch.stack([u_2, u_3], dim=1)

        return all_actions.view(batch_size, num_agents, 2)

    def get_parameters(self) -> Iterator[nn.Parameter]:
        """
        학습 가능한 파라미터 반환.

        Returns:
            네트워크 파라미터들
        """
        yield from self.gat_backbone.parameters()
        yield from self.node_head.parameters()
        yield from self.edge_head.parameters()

    def train(self):
        """학습 모드로 전환."""
        self.gat_backbone.train()
        self.node_head.train()
        self.edge_head.train()

    def eval(self):
        """평가 모드로 전환."""
        self.gat_backbone.eval()
        self.node_head.eval()
        self.edge_head.eval()

    def reset(self) -> None:
        """에피소드 시작 시 컨트롤러 상태 초기화."""
        pass  # 상태 없음
