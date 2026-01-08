"""
학습된 Light-pH 모델 테스트 (시각화 포함).

사용법:
    python test_light_ph.py --model checkpoints/light_ph_final.pt

조작:
    Q: 종료
    R: 리셋
    V: Verbose 토글
    Space: 일시정지/재개
    +/=: 배속 증가
    -: 배속 감소
    1-9: 에이전트 선택 (0: 전체)
    S: 현재 상태 스냅샷 저장 (JSON)
"""

import sys
import argparse
import json
import os
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.colors as mcolors

sys.path.insert(0, '.')

from config import RobotConfig, EnvConfig, GATConfig
from core import MultiAgentEnv
from networks import GATBackbone, NodeHead, EdgeHead
from controller import LightPHController

# 에이전트 색상
AGENT_COLORS = list(mcolors.TABLEAU_COLORS.values())[:10]


def compute_head_outputs(controller, state, goal, neighbors, obstacles=None):
    """헤드 출력값 계산 및 반환."""
    import torch
    from torch_geometric.data import Data

    # 장애물 통합
    all_obstacles = []
    if controller.use_walls:
        wall_positions = controller._get_wall_positions(state.position)
        all_obstacles.extend(wall_positions)
    if obstacles is not None:
        all_obstacles.extend([np.asarray(o) for o in obstacles])

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
        k_g_self = k_g_all[0].item()
        d_self = d_all[0].item()

        # Edge head (ego에서 나가는 엣지)
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
        'k_ij': k_ij,
        'd_ij': d_ij,
        'c_ij': c_ij,
        'k_io': k_io,
        'num_neighbors': num_neighbors,
        'num_obstacles': num_obstacles
    }


def render_simulation(ax, states, goals, trajectories, env_config, robot_config):
    """시뮬레이션 뷰 렌더링."""
    ax.clear()
    ax.set_xlim(-0.1, env_config.width + 0.1)
    ax.set_ylim(-0.1, env_config.height + 0.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)

    # 벽
    wall = Rectangle(
        (0, 0), env_config.width, env_config.height,
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(wall)

    # 궤적
    for i, traj in enumerate(trajectories):
        if len(traj) > 1:
            traj_arr = np.array(traj)
            ax.plot(traj_arr[:, 0], traj_arr[:, 1],
                    color=AGENT_COLORS[i % len(AGENT_COLORS)], alpha=0.5, linewidth=1)

    # 목표
    for i, goal in enumerate(goals):
        ax.plot(goal[0], goal[1], 'x',
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                markersize=12, markeredgewidth=2)

    # 에이전트
    for i, state in enumerate(states):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        circle = Circle(state.position, robot_config.radius,
                       fill=True, facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(circle)
        ax.text(state.position[0], state.position[1] + robot_config.radius + 0.08,
                f'{i}', ha='center', fontsize=9, fontweight='bold')


def save_snapshot(states, goals, head_outputs_list, info, step, episode, env_config, snapshot_dir='snapshots'):
    """
    현재 상태를 JSON으로 저장.

    Args:
        states: 에이전트 상태 리스트
        goals: 목표 위치 배열
        head_outputs_list: 헤드 출력 리스트
        info: 환경 info dict
        step: 현재 스텝
        episode: 현재 에피소드
        env_config: 환경 설정
        snapshot_dir: 저장 디렉토리
    """
    # 디렉토리 생성
    os.makedirs(snapshot_dir, exist_ok=True)

    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{snapshot_dir}/snapshot_ep{episode}_step{step}_{timestamp}.json"

    # 데이터 구성
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'episode': episode,
        'step': step,
        'num_agents': len(states),
        'env': {
            'width': env_config.width,
            'height': env_config.height
        },
        'agents': []
    }

    for i, state in enumerate(states):
        goal = goals[i]
        dist_to_goal = np.linalg.norm(state.position - goal)

        agent_data = {
            'id': i,
            'position': state.position.tolist(),
            'velocity': state.velocity.tolist(),
            'goal': goal.tolist(),
            'dist_to_goal': float(dist_to_goal),
            'arrived': info['arrived'][i] if i < len(info['arrived']) else False
        }

        # 헤드 출력 추가
        if head_outputs_list and i < len(head_outputs_list):
            head_out = head_outputs_list[i]
            agent_data['head_outputs'] = {
                'k_g': float(head_out['k_g']),
                'd_self': float(head_out['d_self']),
                'k_ij': head_out['k_ij'].tolist(),
                'd_ij': head_out['d_ij'].tolist(),
                'c_ij': head_out['c_ij'].tolist(),
                'k_io': head_out['k_io'].tolist(),
                'num_neighbors': head_out['num_neighbors'],
                'num_obstacles': head_out['num_obstacles']
            }

        snapshot['agents'].append(agent_data)

    # 충돌 정보
    snapshot['collisions'] = info.get('collisions', [])

    # 에이전트 간 거리 매트릭스
    n = len(states)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(states[i].position - states[j].position)
            distances.append({
                'agents': [i, j],
                'distance': float(dist)
            })
    snapshot['pairwise_distances'] = distances

    # JSON 저장
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"Snapshot saved: {filename}")
    return filename


def render_stats_panel(ax, head_outputs_list, selected_agent, speed_mult, paused, step, episode):
    """통계 패널 렌더링."""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    y = 0.98
    line_height = 0.035

    # 헤더
    ax.text(0.5, y, "Head Outputs", fontsize=12, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    y -= line_height * 1.5

    # 컨트롤 정보
    status = "PAUSED" if paused else "RUNNING"
    ax.text(0.05, y, f"Speed: {speed_mult:.1f}x | {status}", fontsize=9,
            ha='left', va='top', transform=ax.transAxes,
            color='red' if paused else 'green', fontweight='bold')
    y -= line_height

    ax.text(0.05, y, f"Episode: {episode} | Step: {step}", fontsize=9,
            ha='left', va='top', transform=ax.transAxes)
    y -= line_height * 1.5

    # 선택된 에이전트 표시
    if selected_agent == -1:
        ax.text(0.05, y, "Showing: All Agents", fontsize=9,
                ha='left', va='top', transform=ax.transAxes, fontweight='bold')
    else:
        ax.text(0.05, y, f"Showing: Agent {selected_agent}", fontsize=9,
                ha='left', va='top', transform=ax.transAxes, fontweight='bold',
                color=AGENT_COLORS[selected_agent % len(AGENT_COLORS)])
    y -= line_height * 1.2

    ax.plot([0.05, 0.95], [y + 0.01, y + 0.01], color='gray', linewidth=0.5,
            transform=ax.transAxes)
    y -= line_height * 0.5

    if not head_outputs_list:
        ax.text(0.05, y, "No data (verbose off)", fontsize=9,
                ha='left', va='top', transform=ax.transAxes, style='italic')
        return

    # 에이전트별 출력
    agents_to_show = range(len(head_outputs_list)) if selected_agent == -1 else [selected_agent]

    for i in agents_to_show:
        if i >= len(head_outputs_list):
            continue
        head_out = head_outputs_list[i]
        color = AGENT_COLORS[i % len(AGENT_COLORS)]

        # 에이전트 헤더
        ax.text(0.05, y, f"Agent {i}", fontsize=10, fontweight='bold',
                ha='left', va='top', transform=ax.transAxes, color=color)
        y -= line_height

        # Node head outputs
        ax.text(0.08, y, f"k_g: {head_out['k_g']:.4f}  d_self: {head_out['d_self']:.4f}",
                fontsize=8, ha='left', va='top', transform=ax.transAxes, family='monospace')
        y -= line_height

        # Edge head outputs (neighbors)
        k_ij_str = np.array2string(head_out['k_ij'], precision=3, suppress_small=True, max_line_width=50)
        ax.text(0.08, y, f"k_ij: {k_ij_str}", fontsize=8,
                ha='left', va='top', transform=ax.transAxes, family='monospace')
        y -= line_height

        d_ij_str = np.array2string(head_out['d_ij'], precision=3, suppress_small=True, max_line_width=50)
        ax.text(0.08, y, f"d_ij: {d_ij_str}", fontsize=8,
                ha='left', va='top', transform=ax.transAxes, family='monospace')
        y -= line_height

        c_ij_str = np.array2string(head_out['c_ij'], precision=3, suppress_small=True, max_line_width=50)
        ax.text(0.08, y, f"c_ij: {c_ij_str}", fontsize=8,
                ha='left', va='top', transform=ax.transAxes, family='monospace')
        y -= line_height

        # Obstacle outputs (walls)
        if len(head_out['k_io']) > 0:
            k_io_str = np.array2string(head_out['k_io'], precision=3, suppress_small=True, max_line_width=50)
            ax.text(0.08, y, f"k_io: {k_io_str}", fontsize=8,
                    ha='left', va='top', transform=ax.transAxes, family='monospace', color='brown')
            y -= line_height

        y -= line_height * 0.3  # 에이전트 사이 간격

        if y < 0.05:
            ax.text(0.05, 0.02, "... (press 0-9 to select agent)", fontsize=8,
                    ha='left', va='bottom', transform=ax.transAxes, style='italic')
            break


def main():
    parser = argparse.ArgumentParser(description='Light-pH Test')
    parser.add_argument('--model', type=str, required=True, help='모델 경로')
    parser.add_argument('--num_agents', type=int, default=4, help='에이전트 수')
    parser.add_argument('--device', type=str, default='cuda', help='디바이스')
    parser.add_argument('--no_walls', action='store_true', help='벽을 장애물로 그래프에서 제외 (기본: 포함)')
    parser.add_argument('--verbose', '-v', action='store_true', default=True, help='헤드 출력값 표시 (기본: True)')
    args = parser.parse_args()

    # CUDA 사용 가능 여부 확인
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

    # GAT 네트워크 생성
    gat_backbone = GATBackbone(config=gat_config)
    node_head = NodeHead(input_dim=gat_config.hidden_dim, hidden_dim=32)
    edge_head = EdgeHead(
        input_dim=gat_config.hidden_dim,
        edge_dim=gat_config.edge_input_dim,
        hidden_dim=32
    )

    # Light-pH Controller 생성
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

    # 평가 모드
    controller.eval()

    print(f"=== Light-pH Test ===")
    print(f"Model: {args.model}")
    print(f"Agents: {args.num_agents}")
    print(f"Device: {device}")
    print(f"Use Walls: {not args.no_walls}")
    print("=====================")
    print("Controls:")
    print("  Q: Quit  |  R: Reset  |  V: Toggle verbose")
    print("  Space: Pause/Resume")
    print("  +/=: Speed up  |  -: Slow down")
    print("  0-9: Select agent (0=all)")
    print("  S: Save snapshot (JSON)")
    print("=====================")

    # Figure 생성 (2열 레이아웃)
    fig, (ax_sim, ax_stats) = plt.subplots(1, 2, figsize=(14, 7),
                                           gridspec_kw={'width_ratios': [1.2, 1]})
    fig.suptitle('Light-pH Simulator', fontsize=14, fontweight='bold')

    # 상태 변수
    obs_dict = env.reset()
    trajectories = [[] for _ in range(env_config.num_agents)]

    running = True
    verbose = args.verbose
    paused = False
    speed_mult = 1.0  # 배속 (0.25x ~ 4x)
    selected_agent = -1  # -1 = 전체, 0~N-1 = 특정 에이전트
    step = 0
    episode = 0
    head_outputs_list = []
    info = {'arrived': [False] * args.num_agents, 'collisions': []}

    # 키보드 이벤트 핸들러
    def on_key(event):
        nonlocal running, obs_dict, verbose, paused, speed_mult, selected_agent
        nonlocal trajectories, step, episode, info, head_outputs_list

        if event.key == 'q':
            running = False
        elif event.key == 'r':
            obs_dict = env.reset()
            trajectories = [[] for _ in range(env_config.num_agents)]
            info = {'arrived': [False] * args.num_agents, 'collisions': []}
            head_outputs_list = []
            step = 0
            episode += 1
        elif event.key == 'v':
            verbose = not verbose
            print(f"Verbose: {verbose}")
        elif event.key == ' ':
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif event.key in ['+', '=']:
            speed_mult = min(4.0, speed_mult * 1.5)
            print(f"Speed: {speed_mult:.1f}x")
        elif event.key == '-':
            speed_mult = max(0.25, speed_mult / 1.5)
            print(f"Speed: {speed_mult:.1f}x")
        elif event.key in '0123456789':
            num = int(event.key)
            if num == 0:
                selected_agent = -1
                print("Showing: All agents")
            elif num <= args.num_agents:
                selected_agent = num - 1
                print(f"Showing: Agent {selected_agent}")
        elif event.key == 's':
            # 현재 상태 스냅샷 저장
            save_snapshot(
                states=obs_dict['states'],
                goals=obs_dict['goals'],
                head_outputs_list=head_outputs_list,
                info=info,
                step=step,
                episode=episode,
                env_config=env_config
            )

    fig.canvas.mpl_connect('key_press_event', on_key)

    # 기본 pause 시간 (배속 1x 기준)
    base_pause = 0.02

    while running:
        if not paused:
            states = obs_dict['states']
            goals = obs_dict['goals']

            # 궤적 업데이트
            for i, state in enumerate(states):
                trajectories[i].append(state.position.copy())

            # 각 에이전트 액션 계산
            actions = []
            head_outputs_list = []
            for i in range(args.num_agents):
                state = states[i]
                goal = goals[i]
                neighbors = [states[j] for j in range(args.num_agents) if j != i]

                action = controller.compute_action(state, goal, neighbors)
                actions.append(action)

                # 헤드 출력 계산 (매 스텝)
                if verbose:
                    head_out = compute_head_outputs(controller, state, goal, neighbors)
                    head_outputs_list.append(head_out)

            actions = np.array(actions)

            # 환경 스텝
            obs_dict, rewards, done, info = env.step(actions)

            step += 1

            if done:
                arrived_count = sum(info['arrived'])
                if all(info['arrived']):
                    print(f"Episode {episode}: All agents reached goals in {step} steps!")
                else:
                    print(f"Episode {episode}: Ended at step {step}. "
                          f"Arrived: {arrived_count}/{args.num_agents}")

                plt.pause(1.0 / speed_mult)
                obs_dict = env.reset()
                trajectories = [[] for _ in range(env_config.num_agents)]
                info = {'arrived': [False] * args.num_agents, 'collisions': []}
                head_outputs_list = []
                step = 0
                episode += 1

        # 렌더링 (일시정지 중에도 업데이트)
        render_simulation(ax_sim, obs_dict['states'], obs_dict['goals'],
                         trajectories, env_config, robot_config)

        # 타이틀에 상태 정보
        arrived_count = sum(info['arrived'])
        collision_count = len(info['collisions'])
        ax_sim.set_title(
            f'Episode: {episode} | Step: {step} | '
            f'Arrived: {arrived_count}/{args.num_agents} | '
            f'Collisions: {collision_count}'
        )

        # 통계 패널 렌더링
        render_stats_panel(ax_stats, head_outputs_list, selected_agent,
                          speed_mult, paused, step, episode)

        plt.tight_layout()
        plt.pause(base_pause / speed_mult)

    plt.close(fig)
    print("Test ended.")


if __name__ == "__main__":
    main()
