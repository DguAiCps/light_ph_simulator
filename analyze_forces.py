"""
힘 분석 스크립트.

헤드 출력 (k_g, k_ij, k_io)에 따른 실제 힘 크기를 분석.
모든 힘이 k/r 스케일로 일관됨 (네트워크가 k를 통해 거리 관계 결정)
- Goal attraction: k_g/r * direction
- Neighbor repulsion: k_ij/r * direction
- Wall repulsion: k_io/r * direction

사용법:
    python analyze_forces.py --model checkpoints/light_ph_final.pt
    python analyze_forces.py --model checkpoints/light_ph_final.pt --show_vectors
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import matplotlib.colors as mcolors

sys.path.insert(0, '.')

from config import RobotConfig, EnvConfig, GATConfig
from core import MultiAgentEnv
from networks import GATBackbone, NodeHead, EdgeHead
from controller import LightPHController

AGENT_COLORS = list(mcolors.TABLEAU_COLORS.values())[:10]


def compute_force_components(controller, state, goal, neighbors, neighbor_ids=None):
    """
    각 힘 성분을 분리해서 계산.

    Returns:
        dict: 각 힘 성분과 크기
    """
    from torch_geometric.data import Data

    # 장애물 (벽) 수집
    all_obstacles = []
    if controller.use_walls:
        wall_positions = controller._get_wall_positions(state.position)
        all_obstacles.extend(wall_positions)

    # 그래프 구성
    graph_data = controller._build_graph(state, goal, neighbors, all_obstacles)
    graph_data = graph_data.to(controller.device)

    num_neighbors = graph_data.num_neighbors
    num_obstacles = graph_data.num_obstacles

    with torch.no_grad():
        # GAT forward
        node_embeddings = controller.gat_backbone(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr
        )

        # Node head
        k_g_all, d_all = controller.node_head(node_embeddings)
        k_g = k_g_all[0].item()
        d_self = d_all[0].item()

        # Edge head
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        src_mask = edge_index[0] == 0

        k_ij = []
        d_ij = []
        c_ij = []
        k_io = []

        if src_mask.sum() > 0:
            src_indices = edge_index[0][src_mask]
            dst_indices = edge_index[1][src_mask]

            src_emb = node_embeddings[src_indices]
            dst_emb = node_embeddings[dst_indices]
            edge_feat = edge_attr[src_mask]

            k_all, d_ij_all, c_all = controller.edge_head(src_emb, dst_emb, edge_feat)
            k_all = k_all.cpu().numpy()
            d_ij_all = d_ij_all.cpu().numpy()
            c_all = c_all.cpu().numpy()

            k_ij = k_all[:num_neighbors]
            d_ij = d_ij_all[:num_neighbors]
            c_ij = c_all[:num_neighbors]
            k_io = k_all[num_neighbors:] if num_obstacles > 0 else np.array([])

    # 힘 계산
    q = state.position
    v = state.velocity

    # 1. Goal attraction force: 상수 크기 (거리와 무관)
    # Control law에서 -dH_dq 이므로, goal 방향 인력
    goal_diff = q - goal
    r_goal = np.linalg.norm(goal_diff) + 1e-6
    goal_direction = goal_diff / r_goal
    goal_force = k_g * goal_direction  # 상수 크기: k_g
    goal_force_mag = np.linalg.norm(goal_force)

    # 2. Neighbor repulsion forces (k/r 스케일)
    neighbor_forces = []
    neighbor_details = []
    for j, neighbor in enumerate(neighbors):
        if j < len(k_ij):
            k = k_ij[j]
        else:
            k = 1.0
        diff = q - neighbor.position
        dist = np.linalg.norm(diff) + 1e-6
        force = k * diff / (dist * dist)  # k/r
        force_mag = np.linalg.norm(force)

        # 실제 에이전트 ID 사용 (neighbor_ids가 제공된 경우)
        actual_id = neighbor_ids[j] if neighbor_ids is not None else j

        neighbor_forces.append(force)
        neighbor_details.append({
            'neighbor_idx': actual_id,  # 실제 에이전트 ID
            'k_ij': k,
            'distance': dist,
            'force_mag': force_mag,
            'force_vec': force.tolist()
        })

    total_neighbor_force = np.sum(neighbor_forces, axis=0) if neighbor_forces else np.zeros(2)

    # 3. Obstacle repulsion forces (k/r 스케일, sigmoid 활성화) - wall도 obstacle로 통합
    # LightPHController와 동일한 상수 사용
    OBS_THRESHOLD = 0.1  # 장애물에서 이 거리 이내에서만 반발력 작용 (m)
    OBS_STEEPNESS = 10.0  # sigmoid 전환 급격도

    obstacle_forces = []
    obstacle_details = []
    for o, obs_pos in enumerate(all_obstacles):
        if o < len(k_io):
            k = k_io[o]
        else:
            k = 1.0
        diff = q - obs_pos
        dist = np.linalg.norm(diff) + 1e-6
        # sigmoid 활성화: dist < threshold일 때 ~1, dist > threshold일 때 ~0
        activation = 1.0 / (1.0 + np.exp((dist - OBS_THRESHOLD) * OBS_STEEPNESS))
        force = k * diff / (dist * dist) * activation  # k/r * activation
        force_mag = np.linalg.norm(force)

        obstacle_forces.append(force)
        obstacle_details.append({
            'obs_idx': o,
            'obs_pos': obs_pos.tolist(),
            'k_io': k,
            'distance': dist,
            'force_mag': force_mag,
            'force_vec': force.tolist()
        })

    total_obstacle_force = np.sum(obstacle_forces, axis=0) if obstacle_forces else np.zeros(2)

    # Net dH_dq = goal_force - neighbor_forces - obstacle_forces
    # Net control = -dH_dq = -goal_force + neighbor_forces + obstacle_forces
    net_dH_dq = goal_force - total_neighbor_force - total_obstacle_force

    return {
        'k_g': k_g,
        'd_self': d_self,
        'goal_diff': goal_diff.tolist(),
        'goal_force': goal_force.tolist(),
        'goal_force_mag': goal_force_mag,
        'neighbor_details': neighbor_details,
        'total_neighbor_force': total_neighbor_force.tolist(),
        'total_neighbor_force_mag': np.linalg.norm(total_neighbor_force),
        'obstacle_details': obstacle_details,
        'total_obstacle_force': total_obstacle_force.tolist(),
        'total_obstacle_force_mag': np.linalg.norm(total_obstacle_force),
        'net_dH_dq': net_dH_dq.tolist(),
        'net_dH_dq_mag': np.linalg.norm(net_dH_dq),
        'wall_positions': [o.tolist() for o in all_obstacles]
    }


def draw_force_arrow(ax, start, force_vec, color, scale=0.5, label=None, alpha=0.8, max_length=1.5):
    """힘 벡터 화살표 그리기.

    Args:
        ax: matplotlib axis
        start: 시작점 (agent position)
        force_vec: 힘 벡터
        color: 화살표 색상
        scale: 힘 스케일 (기본 0.5)
        label: 라벨 (optional)
        alpha: 투명도
        max_length: 최대 화살표 길이 (미터) - 이 값을 초과하면 방향 유지하며 클리핑
    """
    force_vec = np.array(force_vec)
    scaled_vec = force_vec * scale
    vec_length = np.linalg.norm(scaled_vec)

    # 최대 길이 초과시 클리핑 (방향 유지)
    if vec_length > max_length and vec_length > 1e-6:
        scaled_vec = scaled_vec * (max_length / vec_length)

    end = start + scaled_vec
    arrow = FancyArrowPatch(start, end, mutation_scale=10,
                            arrowstyle='->', color=color, alpha=alpha, linewidth=2)
    ax.add_patch(arrow)
    if label:
        mid = (start + end) / 2
        ax.text(mid[0], mid[1], label, fontsize=7, color=color)


def main():
    parser = argparse.ArgumentParser(description='Force Analysis')
    parser.add_argument('--model', type=str, required=True, help='모델 경로')
    parser.add_argument('--num_agents', type=int, default=4, help='에이전트 수')
    parser.add_argument('--device', type=str, default='cuda', help='디바이스')
    parser.add_argument('--no_walls', action='store_true', help='벽을 장애물로 그래프에서 제외 (기본: 포함)')
    parser.add_argument('--show_vectors', action='store_true', help='힘 벡터 화살표 표시')
    parser.add_argument('--force_scale', type=float, default=0.1, help='힘 벡터 스케일')
    parser.add_argument('--max_arrow_length', type=float, default=1.5, help='최대 화살표 길이 (m)')
    args = parser.parse_args()

    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # Config
    robot_config = RobotConfig()
    env_config = EnvConfig(num_agents=args.num_agents)
    gat_config = GATConfig()

    # Environment
    env = MultiAgentEnv(env_config, robot_config)

    # Networks
    gat_backbone = GATBackbone(config=gat_config)
    node_head = NodeHead(input_dim=gat_config.hidden_dim, hidden_dim=32)
    edge_head = EdgeHead(
        input_dim=gat_config.hidden_dim,
        edge_dim=gat_config.edge_input_dim,
        hidden_dim=32
    )

    # Controller
    controller = LightPHController(
        gat_backbone=gat_backbone,
        node_head=node_head,
        edge_head=edge_head,
        robot_config=robot_config,
        device=device,
        env_config=env_config if not args.no_walls else None
    )

    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    gat_backbone.load_state_dict(checkpoint['actor_gat'])
    node_head.load_state_dict(checkpoint['actor_node_head'])
    edge_head.load_state_dict(checkpoint['actor_edge_head'])
    controller.eval()

    print("=== Force Analysis ===")
    print(f"Model: {args.model}")
    print(f"Agents: {args.num_agents}")
    print(f"Use Walls: {not args.no_walls}")
    print("======================")
    print("Controls: Q=Quit, R=Reset, Space=Pause, V=Toggle vectors")
    print("======================")

    # Figure setup
    fig, (ax_sim, ax_stats) = plt.subplots(1, 2, figsize=(16, 8),
                                           gridspec_kw={'width_ratios': [1, 1.3]})
    fig.suptitle('Force Analysis - Light-pH', fontsize=14, fontweight='bold')

    # State
    obs_dict = env.reset()
    running = True
    paused = False
    show_vectors = args.show_vectors
    step = 0

    def on_key(event):
        nonlocal running, obs_dict, paused, show_vectors, step
        if event.key == 'q':
            running = False
        elif event.key == 'r':
            obs_dict = env.reset()
            step = 0
        elif event.key == ' ':
            paused = not paused
        elif event.key == 'v':
            show_vectors = not show_vectors
            print(f"Show vectors: {show_vectors}")

    fig.canvas.mpl_connect('key_press_event', on_key)

    while running:
        states = obs_dict['states']
        goals = obs_dict['goals']

        # Force analysis for all agents
        all_force_data = []
        for i in range(args.num_agents):
            state = states[i]
            goal = goals[i]
            neighbor_ids = [j for j in range(args.num_agents) if j != i]
            neighbors = [states[j] for j in neighbor_ids]
            force_data = compute_force_components(controller, state, goal, neighbors, neighbor_ids)
            all_force_data.append(force_data)

        # === Simulation View ===
        ax_sim.clear()
        ax_sim.set_xlim(-0.1, env_config.width + 0.1)
        ax_sim.set_ylim(-0.1, env_config.height + 0.1)
        ax_sim.set_aspect('equal')
        ax_sim.set_xlabel('X (m)')
        ax_sim.set_ylabel('Y (m)')
        ax_sim.grid(True, alpha=0.3)
        ax_sim.set_title(f'Step: {step}')

        # Wall
        wall = Rectangle((0, 0), env_config.width, env_config.height,
                         fill=False, edgecolor='black', linewidth=2)
        ax_sim.add_patch(wall)

        # Goals
        for i, goal in enumerate(goals):
            ax_sim.plot(goal[0], goal[1], 'x',
                       color=AGENT_COLORS[i % len(AGENT_COLORS)],
                       markersize=12, markeredgewidth=2)

        # Agents and force vectors
        for i, state in enumerate(states):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            circle = Circle(state.position, robot_config.radius,
                           fill=True, facecolor=color, edgecolor='black', alpha=0.7)
            ax_sim.add_patch(circle)
            ax_sim.text(state.position[0], state.position[1] + 0.15, f'{i}',
                       ha='center', fontsize=10, fontweight='bold')

            # Force vectors
            if show_vectors:
                fd = all_force_data[i]
                pos = state.position

                # Goal force (red - attraction to goal means force is toward agent)
                if fd['goal_force_mag'] > 0.01:
                    # 실제 힘은 -goal_force 방향 (goal로 끌리는 힘)
                    draw_force_arrow(ax_sim, pos, -np.array(fd['goal_force']),
                                    'red', scale=args.force_scale,
                                    max_length=args.max_arrow_length)

                # Neighbor repulsion (blue)
                if fd['total_neighbor_force_mag'] > 0.01:
                    draw_force_arrow(ax_sim, pos, fd['total_neighbor_force'],
                                    'blue', scale=args.force_scale,
                                    max_length=args.max_arrow_length)

                # Wall repulsion (brown)
                if fd['total_obstacle_force_mag'] > 0.01:
                    draw_force_arrow(ax_sim, pos, fd['total_obstacle_force'],
                                    'brown', scale=args.force_scale,
                                    max_length=args.max_arrow_length)

        # Wall sample points (for visualization)
        if show_vectors and all_force_data[0]['wall_positions']:
            for wp in all_force_data[0]['wall_positions']:
                ax_sim.plot(wp[0], wp[1], 's', color='brown', markersize=4, alpha=0.5)

        # Legend for forces
        if show_vectors:
            ax_sim.plot([], [], '-', color='red', label='Goal attraction', linewidth=2)
            ax_sim.plot([], [], '-', color='blue', label='Neighbor repulsion', linewidth=2)
            ax_sim.plot([], [], '-', color='brown', label='Wall repulsion', linewidth=2)
            ax_sim.legend(loc='upper right', fontsize=8)

        # === Stats Panel ===
        ax_stats.clear()
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis('off')

        y = 0.98
        lh = 0.025  # line height

        ax_stats.text(0.5, y, "Force Analysis", fontsize=12, fontweight='bold',
                     ha='center', transform=ax_stats.transAxes)
        y -= lh * 1.5

        ax_stats.text(0.02, y, "All forces = k/r (일관된 스케일)", fontsize=9, style='italic',
                     transform=ax_stats.transAxes)
        y -= lh
        ax_stats.text(0.02, y, "Goal: attraction | Neighbor/Wall: repulsion",
                     fontsize=8, transform=ax_stats.transAxes)
        y -= lh * 1.5

        # Per-agent breakdown
        for i in range(args.num_agents):
            fd = all_force_data[i]
            color = AGENT_COLORS[i % len(AGENT_COLORS)]

            ax_stats.text(0.02, y, f"━━━ Agent {i} ━━━", fontsize=10, fontweight='bold',
                         color=color, transform=ax_stats.transAxes)
            y -= lh

            # Head parameters
            ax_stats.text(0.04, y, f"k_g={fd['k_g']:.4f}  d_self={fd['d_self']:.4f}",
                         fontsize=8, family='monospace', transform=ax_stats.transAxes)
            y -= lh

            # Goal force
            goal_dist = np.linalg.norm(fd['goal_diff'])
            ax_stats.text(0.04, y, f"Goal: dist={goal_dist:.3f}m → force={fd['goal_force_mag']:.4f}N",
                         fontsize=8, family='monospace', transform=ax_stats.transAxes, color='darkred')
            y -= lh

            # Neighbor forces
            ax_stats.text(0.04, y, f"Neighbors ({len(fd['neighbor_details'])}):",
                         fontsize=8, family='monospace', transform=ax_stats.transAxes, color='darkblue')
            y -= lh
            for nd in fd['neighbor_details']:
                ax_stats.text(0.06, y, f"  #{nd['neighbor_idx']}: k={nd['k_ij']:.3f}, "
                             f"dist={nd['distance']:.3f}m → F={nd['force_mag']:.4f}N",
                             fontsize=7, family='monospace', transform=ax_stats.transAxes)
                y -= lh * 0.9
            ax_stats.text(0.06, y, f"  Total: {fd['total_neighbor_force_mag']:.4f}N",
                         fontsize=8, family='monospace', transform=ax_stats.transAxes,
                         fontweight='bold')
            y -= lh

            # Obstacle forces
            if fd['obstacle_details']:
                ax_stats.text(0.04, y, f"Walls ({len(fd['obstacle_details'])}):",
                             fontsize=8, family='monospace', transform=ax_stats.transAxes, color='brown')
                y -= lh
                for od in fd['obstacle_details']:
                    ax_stats.text(0.06, y, f"  wall: k={od['k_io']:.3f}, "
                                 f"dist={od['distance']:.3f}m → F={od['force_mag']:.4f}N",
                                 fontsize=7, family='monospace', transform=ax_stats.transAxes)
                    y -= lh * 0.9
                ax_stats.text(0.06, y, f"  Total: {fd['total_obstacle_force_mag']:.4f}N",
                             fontsize=8, family='monospace', transform=ax_stats.transAxes,
                             fontweight='bold')
                y -= lh

            # Net force
            ax_stats.text(0.04, y, f"Net |dH/dq|: {fd['net_dH_dq_mag']:.4f}",
                         fontsize=8, family='monospace', transform=ax_stats.transAxes,
                         fontweight='bold', color='black')
            y -= lh * 1.5

            if y < 0.05:
                ax_stats.text(0.02, 0.02, "... (more agents below)", fontsize=8,
                             style='italic', transform=ax_stats.transAxes)
                break

        plt.tight_layout()

        if not paused:
            # Step simulation
            actions = []
            for i in range(args.num_agents):
                state = states[i]
                goal = goals[i]
                neighbors = [states[j] for j in range(args.num_agents) if j != i]
                action = controller.compute_action(state, goal, neighbors)
                actions.append(action)
            actions = np.array(actions)
            obs_dict, rewards, done, info = env.step(actions)
            step += 1

            if done:
                print(f"Episode ended at step {step}. Arrived: {info['arrived']}")
                obs_dict = env.reset()
                step = 0

        plt.pause(0.05)

    plt.close(fig)
    print("Analysis ended.")


if __name__ == "__main__":
    main()
