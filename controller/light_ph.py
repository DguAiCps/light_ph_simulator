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
    from ..config import RobotConfig, GATConfig, EnvConfig
    from ..networks import GATBackbone, NodeHead, EdgeHead
except ImportError:
    from config import RobotConfig, GATConfig, EnvConfig
    from networks import GATBackbone, NodeHead, EdgeHead


class LightPHController(BaseController):
    """
    Light-pH 컨트롤러.

    BaseController를 상속. 논문의 핵심 구현.
    GAT로 이웃 정보를 처리하고 Port-Hamiltonian 제어 법칙 적용.
    """

    # 장애물 반발력 활성화 파라미터 (고정값) - wall도 obstacle로 통합
    OBS_THRESHOLD = 0.1  # 장애물에서 이 거리 이내에서만 반발력 작용 (m)
    OBS_STEEPNESS = 10.0  # sigmoid 전환 급격도

    def __init__(
        self,
        gat_backbone: GATBackbone,
        node_head: NodeHead,
        edge_head: EdgeHead,
        robot_config: RobotConfig,
        device: str = 'cuda',
        env_config: EnvConfig = None
    ):
        """
        Args:
            gat_backbone: GAT 네트워크
            node_head: 노드 스칼라 출력 헤드
            edge_head: 엣지 스칼라 출력 헤드
            robot_config: 로봇 파라미터
            device: 디바이스
            env_config: 환경 설정 (벽 경계 정보)
        """
        self.gat_backbone = gat_backbone
        self.node_head = node_head
        self.edge_head = edge_head
        self.robot_config = robot_config
        self.device = device
        self.env_config = env_config
        self.use_walls = env_config is not None

        # 모델을 디바이스로 이동
        self.gat_backbone.to(device)
        self.node_head.to(device)
        self.edge_head.to(device)

    def compute_action(
        self,
        state: 'AgentState',
        goal: np.ndarray,
        neighbors: List['AgentState'],
        obstacles: List[np.ndarray] = None
    ) -> np.ndarray:
        """
        Light-pH 제어 입력 계산.

        Args:
            state: 현재 상태
            goal: 목표 위치
            neighbors: 이웃들
            obstacles: 장애물 위치 리스트 [(x, y), ...] - None이면 벽만 사용

        Returns:
            힘 [Fx, Fy]
        """
        # 장애물 통합: 벽 + 외부 장애물
        all_obstacles = []
        if self.use_walls:
            wall_positions = self._get_wall_positions(state.position)
            all_obstacles.extend(wall_positions)
        if obstacles is not None:
            all_obstacles.extend([np.asarray(o) for o in obstacles])

        # 1. 그래프 구성
        graph_data = self._build_graph(state, goal, neighbors, all_obstacles)
        graph_data = graph_data.to(self.device)

        num_neighbors = graph_data.num_neighbors
        num_obstacles = graph_data.num_obstacles

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

                k_all, d_all, c_all = self.edge_head(src_emb, dst_emb, edge_feat)
                k_all = k_all.cpu().numpy()
                d_all = d_all.cpu().numpy()
                c_all = c_all.cpu().numpy()

                # 이웃 엣지와 장애물 엣지 분리
                # ego 에서 나가는 엣지 순서: neighbors (num_neighbors) -> obstacles (num_obstacles)
                k_ij = k_all[:num_neighbors]
                d_ij = d_all[:num_neighbors]
                c_ij = c_all[:num_neighbors]

                if num_obstacles > 0:
                    k_io = k_all[num_neighbors:num_neighbors + num_obstacles]
                else:
                    k_io = np.array([])
            else:
                k_ij = np.array([])
                d_ij = np.array([])
                c_ij = np.array([])
                k_io = np.array([])

        # 4. Analytic ∇H 계산
        grad_H = self._compute_hamiltonian_gradient(
            state, goal, neighbors, k_ij, k_g_self,
            obstacle_positions=all_obstacles if len(all_obstacles) > 0 else None,
            k_io=k_io if len(k_io) > 0 else None
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
        neighbors: List['AgentState'],
        obstacles: List[np.ndarray] = None
    ) -> Data:
        """
        torch_geometric용 그래프 데이터 생성.

        Args:
            state: 자기 상태
            goal: 목표 위치
            neighbors: 이웃들
            obstacles: 장애물 위치 리스트 [(x, y), ...]

        Returns:
            torch_geometric.data.Data: 그래프 데이터
        """
        num_neighbors = len(neighbors)
        obstacles = obstacles if obstacles is not None else []
        num_obstacles = len(obstacles)
        num_nodes = 1 + num_neighbors + num_obstacles  # 자기 + 이웃들 + 장애물들

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

        # 장애물 노드들 (obstacle type)
        for obs_pos in obstacles:
            obs_type = np.array([0, 0, 1])  # obstacle
            obs_state = np.concatenate([obs_pos, np.zeros(2)])  # 위치 + 속도 0
            obs_mission = np.array([0, 0, 0])  # 해당 없음
            obs_goal_offset = np.array([0, 0])  # 해당 없음
            obs_node = np.concatenate([
                obs_type, obs_state, obs_mission, obs_goal_offset
            ])
            node_features.append(obs_node)

        x = torch.tensor(np.array(node_features), dtype=torch.float32)

        # Edge features: [Δq(2), Δv(2), r(1), q̂(2)] = 7
        edge_index = []
        edge_attr = []

        # 에이전트들 (자기 + 이웃들) - fully connected
        all_agent_states = [state] + neighbors
        num_agents = 1 + num_neighbors

        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    edge_index.append([i, j])

                    src_state = all_agent_states[i]
                    dst_state = all_agent_states[j]

                    delta_q = dst_state.position - src_state.position
                    delta_v = dst_state.velocity - src_state.velocity
                    r = np.linalg.norm(delta_q)
                    q_hat = delta_q / (r + 1e-6)

                    edge_feat = np.concatenate([delta_q, delta_v, [r], q_hat])
                    edge_attr.append(edge_feat)

        # 에이전트 → 장애물 엣지 (자기 노드에서만)
        for obs_idx, obs_pos in enumerate(obstacles):
            obs_node_idx = num_agents + obs_idx
            # Ego (자기 노드) → Obstacle
            edge_index.append([0, obs_node_idx])

            delta_q = obs_pos - state.position
            delta_v = np.zeros(2) - state.velocity  # 장애물 속도 = 0
            r = np.linalg.norm(delta_q)
            q_hat = delta_q / (r + 1e-6)

            edge_feat = np.concatenate([delta_q, delta_v, [r], q_hat])
            edge_attr.append(edge_feat)

        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # (2, E)
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float32)  # (E, 7)

        # 메타 정보 저장 (나중에 사용)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_obstacles = num_obstacles
        data.num_neighbors = num_neighbors

        return data

    def _compute_hamiltonian_gradient(
        self,
        state: 'AgentState',
        goal: np.ndarray,
        neighbors: List['AgentState'],
        k: np.ndarray,
        k_g: float,
        obstacle_positions: List[np.ndarray] = None,
        k_io: np.ndarray = None
    ) -> np.ndarray:
        """
        Analytic Hamiltonian gradient 계산 (논문 eq.9).

        Args:
            state: 현재 상태
            goal: 목표 위치
            neighbors: 이웃들
            k: 에이전트 간 스프링 상수 (len = num_neighbors)
            k_g: 목표 스프링 상수
            obstacle_positions: 장애물 위치들 (벽 포함)
            k_io: 장애물 상호작용 스프링 상수 (len = num_obstacles)

        Returns:
            [∂H/∂q_x, ∂H/∂q_y, ∂H/∂v_x, ∂H/∂v_y]
        """
        q = state.position
        v = state.velocity
        m = self.robot_config.mass

        # Goal: 상수 크기 (거리와 무관하게 일정한 인력)
        # Neighbor/Obstacle: k/r 스케일 (가까울수록 강한 척력)
        goal_diff = q - goal
        r_goal = np.linalg.norm(goal_diff) + 1e-6
        goal_direction = goal_diff / r_goal
        dH_dq = k_g * goal_direction  # 상수 크기: k_g

        # 이웃 척력 (Top-k: k_ij가 높은 상위 N개만 고려)
        top_k_neighbors = 2  # 상위 2개 이웃만
        if len(neighbors) > 0:
            # 각 이웃의 (k_ij, force) 계산
            neighbor_forces = []
            for j, neighbor in enumerate(neighbors):
                if j < len(k):
                    k_ij = k[j]
                else:
                    k_ij = 1.0
                diff = q - neighbor.position
                r = np.linalg.norm(diff) + 1e-6
                force = k_ij * diff / (r * r)  # k_ij/r
                neighbor_forces.append((k_ij, force))

            # k_ij 기준 내림차순 정렬 후 상위 N개 선택
            neighbor_forces.sort(key=lambda x: x[0], reverse=True)
            top_k = min(top_k_neighbors, len(neighbor_forces))

            # 상위 N개 힘 합산 후 N으로 나눔
            neighbor_sum = np.zeros(2)
            for i in range(top_k):
                neighbor_sum += neighbor_forces[i][1]
            dH_dq -= neighbor_sum / top_k

        # 장애물 척력 (sigmoid 활성화: threshold 이내에서만 작용)
        if obstacle_positions is not None and k_io is not None:
            for o, obs_pos in enumerate(obstacle_positions):
                if o < len(k_io):
                    k_io_val = k_io[o]
                else:
                    k_io_val = 1.0
                diff = q - obs_pos
                r = np.linalg.norm(diff) + 1e-6
                # sigmoid 활성화: r < threshold일 때 ~1, r > threshold일 때 ~0
                activation = 1.0 / (1.0 + np.exp((r - self.OBS_THRESHOLD) * self.OBS_STEEPNESS))
                dH_dq -= k_io_val * diff / (r * r) * activation  # k_io/r * activation

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

    def _get_wall_positions(self, agent_pos: np.ndarray) -> List[np.ndarray]:
        """
        각 벽에서 에이전트에 가장 가까운 점 계산.

        Args:
            agent_pos: 에이전트 위치 [x, y]

        Returns:
            4개 벽의 가장 가까운 점 리스트
            [left_wall, right_wall, bottom_wall, top_wall]
        """
        if self.env_config is None:
            return []

        x, y = agent_pos
        w, h = self.env_config.width, self.env_config.height

        # 각 벽에서 가장 가까운 점
        wall_positions = [
            np.array([0.0, np.clip(y, 0, h)]),      # Left wall
            np.array([w, np.clip(y, 0, h)]),        # Right wall
            np.array([np.clip(x, 0, w), 0.0]),      # Bottom wall
            np.array([np.clip(x, 0, w), h]),        # Top wall
        ]

        return wall_positions

    def forward_differentiable(
        self,
        states_batch: List[dict],
        num_agents: int
    ) -> torch.Tensor:
        """
        Fully vectorized differentiable forward pass.

        Args:
            states_batch: 배치된 상태 딕셔너리 리스트
                - 'states': 에이전트 상태들
                - 'goals': 목표 위치들
                - 'obstacles': (선택) 장애물 위치들 [(x, y), ...]
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
        all_obstacle_q = []  # 장애물 위치들
        has_obstacles = False

        for state_dict in states_batch:
            agent_states = state_dict['states']
            goals = state_dict['goals']

            # 장애물 수집 (벽 + 외부 장애물)
            obstacles = []
            if self.use_walls:
                # 각 에이전트별 벽 위치는 에이전트 위치에 따라 다름
                pass  # 아래에서 에이전트별로 계산
            if 'obstacles' in state_dict and state_dict['obstacles'] is not None:
                obstacles.extend([np.asarray(o) for o in state_dict['obstacles']])
                has_obstacles = True

            for i in range(num_agents):
                state = agent_states[i]
                goal = goals[i]
                neighbors = [agent_states[j] for j in range(num_agents) if j != i]

                # 에이전트별 장애물 (벽 + 공통 장애물)
                agent_obstacles = []
                if self.use_walls:
                    wall_pos = self._get_wall_positions(state.position)
                    agent_obstacles.extend(wall_pos)
                    has_obstacles = True
                agent_obstacles.extend(obstacles)

                graph_data = self._build_graph(state, goal, neighbors, agent_obstacles)
                all_graphs.append(graph_data)

                all_q.append(state.position)
                all_v.append(state.velocity)
                all_g.append(goal)
                all_neighbor_q.append([n.position for n in neighbors])
                all_obstacle_q.append([o for o in agent_obstacles])

        # 텐서 변환
        q_tensor = torch.tensor(np.array(all_q), dtype=torch.float32, device=device)
        v_tensor = torch.tensor(np.array(all_v), dtype=torch.float32, device=device)
        g_tensor = torch.tensor(np.array(all_g), dtype=torch.float32, device=device)
        neighbor_q_tensor = torch.tensor(np.array(all_neighbor_q), dtype=torch.float32, device=device)
        m = self.robot_config.mass

        # 장애물 수 (모두 동일하다고 가정)
        num_obstacles = len(all_obstacle_q[0]) if all_obstacle_q and len(all_obstacle_q[0]) > 0 else 0

        if num_obstacles > 0:
            obstacle_q_tensor = torch.tensor(np.array(all_obstacle_q), dtype=torch.float32, device=device)

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

        # ========== 4. Edge head (Ego에서 나가는 모든 엣지) ==========
        src, dst = batched_graph.edge_index

        is_ego = torch.zeros(all_node_embeddings.size(0), dtype=torch.bool, device=device)
        is_ego[ego_indices] = True
        ego_edge_mask = is_ego[src]

        ego_src = src[ego_edge_mask]
        ego_dst = dst[ego_edge_mask]
        ego_edge_attr = batched_graph.edge_attr[ego_edge_mask]

        k_all, d_all, c_all = self.edge_head(
            all_node_embeddings[ego_src],
            all_node_embeddings[ego_dst],
            ego_edge_attr
        )

        # 엣지 분리: neighbors (num_neighbors) + obstacles (num_obstacles)
        num_ego_edges = num_neighbors + num_obstacles
        k_all_matrix = k_all.view(total_graphs, num_ego_edges)
        d_all_matrix = d_all.view(total_graphs, num_ego_edges)
        c_all_matrix = c_all.view(total_graphs, num_ego_edges)

        k_ij_matrix = k_all_matrix[:, :num_neighbors]
        d_ij_matrix = d_all_matrix[:, :num_neighbors]
        c_ij_matrix = c_all_matrix[:, :num_neighbors]

        # ========== 5. Vectorized Interaction Force (Neighbors, Top-k) ==========
        top_k_neighbors = 2  # 상위 2개 이웃만 고려
        q_diff = q_tensor.unsqueeze(1) - neighbor_q_tensor  # (B*N, num_neighbors, 2)
        r_neighbor = torch.norm(q_diff, dim=-1, keepdim=True) + 1e-6  # (B*N, num_neighbors, 1)
        # k/r 스케일: diff / r^2 = direction / r
        weighted_diff = k_ij_matrix.unsqueeze(-1) * q_diff / (r_neighbor * r_neighbor)  # (B*N, num_neighbors, 2)

        # Top-k: k_ij가 높은 상위 N개만 선택
        actual_k = min(top_k_neighbors, num_neighbors)
        _, top_indices = torch.topk(k_ij_matrix, actual_k, dim=1)  # (B*N, actual_k)

        # 상위 k개 힘만 선택하여 합산
        batch_indices = torch.arange(total_graphs, device=device).unsqueeze(1).expand(-1, actual_k)
        top_weighted_diff = weighted_diff[batch_indices, top_indices]  # (B*N, actual_k, 2)
        interaction_sum = top_weighted_diff.sum(dim=1) / actual_k  # N으로 나눔

        # ========== 5b. Obstacle Interaction Force (sigmoid 활성화) ==========
        if num_obstacles > 0:
            k_io_matrix = k_all_matrix[:, num_neighbors:num_neighbors + num_obstacles]
            obs_q_diff = q_tensor.unsqueeze(1) - obstacle_q_tensor  # (B*N, num_obstacles, 2)
            r_obs = torch.norm(obs_q_diff, dim=-1, keepdim=True) + 1e-6  # (B*N, num_obstacles, 1)
            # sigmoid 활성화: r < threshold일 때 ~1, r > threshold일 때 ~0
            activation = torch.sigmoid((self.OBS_THRESHOLD - r_obs.squeeze(-1)) * self.OBS_STEEPNESS)  # (B*N, num_obstacles)
            obs_weighted_diff = k_io_matrix.unsqueeze(-1) * obs_q_diff / (r_obs * r_obs) * activation.unsqueeze(-1)
            obstacle_sum = obs_weighted_diff.sum(dim=1)
            interaction_sum = interaction_sum + obstacle_sum

        # ========== 6. Hamiltonian Gradient ==========
        # Goal: 상수 크기 (거리와 무관하게 일정한 인력)
        goal_diff = q_tensor - g_tensor  # (B*N, 2)
        r_goal = torch.norm(goal_diff, dim=-1, keepdim=True) + 1e-6  # (B*N, 1)
        goal_direction = goal_diff / r_goal  # (B*N, 2)
        goal_term = k_g_ego * goal_direction  # 상수 크기: k_g
        dH_dq = goal_term - interaction_sum
        dH_dv = m * v_tensor

        # ========== 7. Vectorized d_avg, c_avg ==========
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
