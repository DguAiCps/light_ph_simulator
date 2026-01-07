"""
학습된 Light-pH 모델 테스트 (시각화 포함).

사용법:
    python test_light_ph.py --model checkpoints/light_ph_final.pt
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from config import RobotConfig, EnvConfig, GATConfig
from core import MultiAgentEnv
from networks import GATBackbone, NodeHead, EdgeHead
from controller import LightPHController
from utils.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description='Light-pH Test')
    parser.add_argument('--model', type=str, required=True, help='모델 경로')
    parser.add_argument('--num_agents', type=int, default=4, help='에이전트 수')
    parser.add_argument('--device', type=str, default='cuda', help='디바이스')
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

    # 환경 및 시각화 생성
    env = MultiAgentEnv(env_config, robot_config)
    vis = Visualizer(env_config, robot_config)

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
        device=device
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
    print("Press Q to quit, R to reset")
    print("=====================")

    # 환경 초기화
    obs_dict = env.reset()
    vis.reset()

    running = True

    def on_key(event):
        nonlocal running, obs_dict
        if event.key == 'q':
            running = False
        elif event.key == 'r':
            obs_dict = env.reset()
            vis.trajectories = [[] for _ in range(env_config.num_agents)]

    vis.fig.canvas.mpl_connect('key_press_event', on_key)

    step = 0
    episode = 0

    while running:
        states = obs_dict['states']
        goals = obs_dict['goals']

        # 각 에이전트 액션 계산
        actions = []
        for i in range(args.num_agents):
            state = states[i]
            goal = goals[i]
            neighbors = [states[j] for j in range(args.num_agents) if j != i]

            action = controller.compute_action(state, goal, neighbors)
            actions.append(action)

        actions = np.array(actions)

        # 환경 스텝
        obs_dict, rewards, done, info = env.step(actions)

        # 벽 충돌 검사
        wall_contact_dist = robot_config.radius + 0.01
        for i, state in enumerate(obs_dict['states']):
            pos = state.position
            if (pos[0] < wall_contact_dist or
                pos[0] > env_config.width - wall_contact_dist or
                pos[1] < wall_contact_dist or
                pos[1] > env_config.height - wall_contact_dist):
                print(f"crash! (agent {i})")

        # 시각화
        vis.render(obs_dict['states'], obs_dict['goals'], show_trajectory=True)

        # 상태 표시
        arrived_count = sum(info['arrived'])
        collision_count = len(info['collisions'])
        vis.ax.set_title(
            f'Light-pH | Episode: {episode} | Step: {step} | '
            f'Arrived: {arrived_count}/{args.num_agents} | '
            f'Collisions: {collision_count}'
        )

        step += 1

        if done:
            if all(info['arrived']):
                print(f"Episode {episode}: All agents reached goals in {step} steps!")
            else:
                print(f"Episode {episode}: Ended at step {step}. "
                      f"Arrived: {arrived_count}/{args.num_agents}")

            plt.pause(1.0)
            obs_dict = env.reset()
            vis.trajectories = [[] for _ in range(env_config.num_agents)]
            step = 0
            episode += 1

        plt.pause(0.02)

    vis.close()
    print("Test ended.")


if __name__ == "__main__":
    main()
