"""
에피소드 전체를 JSON으로 기록.

사용법:
    python record_episode.py --model checkpoints/light_ph_final.pt
    python record_episode.py --model checkpoints/light_ph_final.pt --no_render
"""

import sys
import argparse
import json
import os
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, '.')

from config import RobotConfig, EnvConfig, GATConfig
from core import MultiAgentEnv
from networks import GATBackbone, NodeHead, EdgeHead
from controller import LightPHController


def compute_head_outputs(controller, state, goal, neighbors):
    """헤드 출력값 계산."""
    from torch_geometric.data import Data

    all_obstacles = []
    if controller.use_walls:
        wall_positions = controller._get_wall_positions(state.position)
        all_obstacles.extend(wall_positions)

    graph_data = controller._build_graph(state, goal, neighbors, all_obstacles)
    graph_data = graph_data.to(controller.device)

    num_neighbors = graph_data.num_neighbors
    num_obstacles = graph_data.num_obstacles

    with torch.no_grad():
        node_embeddings = controller.gat_backbone(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr
        )

        k_g_all, d_all = controller.node_head(node_embeddings)
        k_g_self = k_g_all[0].item()
        d_self = d_all[0].item()

        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        src_mask = edge_index[0] == 0

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
        else:
            k_ij = np.array([])
            d_ij = np.array([])
            c_ij = np.array([])
            k_io = np.array([])

    return {
        'k_g': k_g_self,
        'd_self': d_self,
        'k_ij': k_ij.tolist(),
        'd_ij': d_ij.tolist(),
        'c_ij': c_ij.tolist(),
        'k_io': k_io.tolist(),
        'num_neighbors': num_neighbors,
        'num_obstacles': num_obstacles
    }


def record_episode(env, controller, num_agents, render=False):
    """에피소드 전체 기록."""
    if render:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        import matplotlib.colors as mcolors
        AGENT_COLORS = list(mcolors.TABLEAU_COLORS.values())[:10]

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.ion()

    obs_dict = env.reset()

    # 기록 데이터
    episode_data = {
        'timestamp': datetime.now().isoformat(),
        'num_agents': num_agents,
        'env': {
            'width': env.env_config.width,
            'height': env.env_config.height
        },
        'initial_positions': [s.position.tolist() for s in obs_dict['states']],
        'goals': obs_dict['goals'].tolist(),
        'steps': []
    }

    done = False
    step = 0

    while not done:
        states = obs_dict['states']
        goals = obs_dict['goals']

        # 스텝 데이터
        step_data = {
            'step': step,
            'agents': []
        }

        actions = []
        for i in range(num_agents):
            state = states[i]
            goal = goals[i]
            neighbors = [states[j] for j in range(num_agents) if j != i]

            # 액션 계산
            action = controller.compute_action(state, goal, neighbors)
            actions.append(action)

            # 헤드 출력 계산
            head_out = compute_head_outputs(controller, state, goal, neighbors)

            # 에이전트 데이터
            dist_to_goal = np.linalg.norm(state.position - goal)
            agent_data = {
                'id': i,
                'position': state.position.tolist(),
                'velocity': state.velocity.tolist(),
                'dist_to_goal': float(dist_to_goal),
                'action': action.tolist(),
                'head_outputs': head_out
            }
            step_data['agents'].append(agent_data)

        # 에이전트 간 거리
        pairwise_dist = []
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = np.linalg.norm(states[i].position - states[j].position)
                pairwise_dist.append({
                    'pair': [i, j],
                    'distance': float(dist)
                })
        step_data['pairwise_distances'] = pairwise_dist

        episode_data['steps'].append(step_data)

        # 환경 스텝
        actions = np.array(actions)
        obs_dict, rewards, done, info = env.step(actions)

        # 렌더링
        if render:
            ax.clear()
            ax.set_xlim(-0.1, env.env_config.width + 0.1)
            ax.set_ylim(-0.1, env.env_config.height + 0.1)
            ax.set_aspect('equal')
            ax.set_title(f'Step {step}')
            ax.grid(True, alpha=0.3)

            # 벽
            wall = Rectangle((0, 0), env.env_config.width, env.env_config.height,
                            fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(wall)

            # 목표
            for i, goal in enumerate(goals):
                ax.plot(goal[0], goal[1], 'x', color=AGENT_COLORS[i % len(AGENT_COLORS)],
                       markersize=12, markeredgewidth=2)

            # 에이전트
            for i, state in enumerate(states):
                color = AGENT_COLORS[i % len(AGENT_COLORS)]
                circle = Circle(state.position, env.robot_config.radius,
                              fill=True, facecolor=color, edgecolor='black', alpha=0.7)
                ax.add_patch(circle)
                ax.text(state.position[0], state.position[1] + 0.15, f'{i}',
                       ha='center', fontsize=10, fontweight='bold')

            plt.pause(0.01)

        step += 1

    # 결과 정보
    episode_data['total_steps'] = step
    episode_data['result'] = {
        'arrived': info['arrived'],
        'all_arrived': all(info['arrived']),
        'collisions': info.get('collisions', [])
    }

    if render:
        plt.ioff()
        plt.close(fig)

    return episode_data


def main():
    parser = argparse.ArgumentParser(description='Record Episode')
    parser.add_argument('--model', type=str, required=True, help='모델 경로')
    parser.add_argument('--num_agents', type=int, default=4, help='에이전트 수')
    parser.add_argument('--device', type=str, default='cuda', help='디바이스')
    parser.add_argument('--no_walls', action='store_true', help='벽을 장애물로 그래프에서 제외 (기본: 포함)')
    parser.add_argument('--output_dir', type=str, default='recordings', help='저장 디렉토리')
    parser.add_argument('--no_render', action='store_true', help='렌더링 끄기')
    args = parser.parse_args()

    # 디바이스 설정
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # 설정 생성
    robot_config = RobotConfig()
    env_config = EnvConfig(num_agents=args.num_agents)
    gat_config = GATConfig()

    # 환경 생성
    env = MultiAgentEnv(env_config, robot_config)

    # 네트워크 생성
    gat_backbone = GATBackbone(config=gat_config)
    node_head = NodeHead(input_dim=gat_config.hidden_dim, hidden_dim=32)
    edge_head = EdgeHead(
        input_dim=gat_config.hidden_dim,
        edge_dim=gat_config.edge_input_dim,
        hidden_dim=32
    )

    # 컨트롤러 생성
    controller = LightPHController(
        gat_backbone=gat_backbone,
        node_head=node_head,
        edge_head=edge_head,
        robot_config=robot_config,
        device=device,
        env_config=env_config if not args.no_walls else None
    )

    # 모델 로드
    checkpoint = torch.load(args.model, map_location=device)
    gat_backbone.load_state_dict(checkpoint['actor_gat'])
    node_head.load_state_dict(checkpoint['actor_node_head'])
    edge_head.load_state_dict(checkpoint['actor_edge_head'])
    controller.eval()

    print(f"=== Recording Episode ===")
    print(f"Model: {args.model}")
    print(f"Agents: {args.num_agents}")
    print(f"Use Walls: {not args.no_walls}")
    print("=========================")

    # 에피소드 기록
    episode_data = record_episode(
        env, controller, args.num_agents,
        render=not args.no_render
    )

    # 저장
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{args.output_dir}/episode_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(episode_data, f, indent=2, ensure_ascii=False)

    print(f"\nEpisode recorded: {filename}")
    print(f"Total steps: {episode_data['total_steps']}")
    print(f"All arrived: {episode_data['result']['all_arrived']}")
    print(f"Arrived: {episode_data['result']['arrived']}")


if __name__ == "__main__":
    main()
