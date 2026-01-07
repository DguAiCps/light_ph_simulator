"""
학습된 MAPPO 모델 테스트 (시각화 포함).

사용법:
    python test_mappo.py --model checkpoints/mappo_final.pt
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from config import RobotConfig, EnvConfig
from core import MultiAgentEnv
from networks import Actor
from utils.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description='MAPPO Test')
    parser.add_argument('--model', type=str, required=True, help='모델 경로')
    parser.add_argument('--num_agents', type=int, default=4, help='에이전트 수')
    parser.add_argument('--deterministic', action='store_true', help='결정적 행동')
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

    # 환경 및 시각화 생성
    env = MultiAgentEnv(env_config, robot_config)
    vis = Visualizer(env_config, robot_config)

    # Observation 차원 계산
    obs_dim = 4 + 2 + 4 * (args.num_agents - 1)
    action_dim = 2

    # Actor 로드
    actor = Actor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()

    print(f"=== MAPPO Test ===")
    print(f"Model: {args.model}")
    print(f"Agents: {args.num_agents}")
    print(f"Device: {device}")
    print(f"Deterministic: {args.deterministic}")
    print("Press Q to quit, R to reset")
    print("==================")

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
        # Observation 추출
        obs = np.array([env.get_observation(i) for i in range(args.num_agents)])
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        # 액션 선택
        with torch.no_grad():
            actions, _, _ = actor.get_action(obs_tensor, deterministic=args.deterministic)
            actions = actions.cpu().numpy()

        # 환경 스텝
        obs_dict, rewards, done, info = env.step(actions)

        # 시각화
        vis.render(obs_dict['states'], obs_dict['goals'], show_trajectory=True)

        # 상태 표시
        arrived_count = sum(info['arrived'])
        collision_count = len(info['collisions'])
        vis.ax.set_title(
            f'MAPPO | Episode: {episode} | Step: {step} | '
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
