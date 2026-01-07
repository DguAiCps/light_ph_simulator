"""
Light-pH Simulator 진입점.

Light-pH 논문 검증을 위한 2D 멀티 에이전트 시뮬레이터.

사용법:
    # 학습
    python main.py --mode train --episodes 1000

    # 평가
    python main.py --mode eval --checkpoint checkpoints/light_ph_final.pt --visualize

    # 설정 파일 지정
    python main.py --config my_config.yaml --mode train
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from config import (
    RobotConfig, EnvConfig, GATConfig, TrainingConfig,
    load_config, save_config
)
from core import MultiAgentEnv
from networks import GATBackbone, NodeHead, EdgeHead
from controller import LightPHController
from training import ReplayBuffer, CentralizedCritic, SACAgent, Trainer
from utils.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description='Light-pH Simulator')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='설정 파일 경로')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'],
                        help='실행 모드 (train/eval)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='로드할 체크포인트')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='에피소드 수')
    parser.add_argument('--visualize', action='store_true',
                        help='시각화 여부')
    parser.add_argument('--device', type=str, default='cuda',
                        help='디바이스')
    parser.add_argument('--save_path', type=str, default='checkpoints',
                        help='체크포인트 저장 경로')
    args = parser.parse_args()

    # 디바이스 설정
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # 1. Config 로드
    try:
        config_dict = load_config(args.config)
        robot_config = RobotConfig(**config_dict.get('robot', {}))
        env_config = EnvConfig(**config_dict.get('environment', {}))
        gat_config = GATConfig(**config_dict.get('gat', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        print(f"Config loaded from {args.config}")
    except FileNotFoundError:
        print(f"Config file not found: {args.config}. Using defaults.")
        robot_config = RobotConfig()
        env_config = EnvConfig()
        gat_config = GATConfig()
        training_config = TrainingConfig()

    print("=== Light-pH Simulator ===")
    print(f"Mode: {args.mode}")
    print(f"Agents: {env_config.num_agents}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {device}")
    print("==========================")

    # 2. Environment 생성
    env = MultiAgentEnv(env_config, robot_config)

    # 3. Networks 생성 (GATBackbone, Heads)
    gat_backbone = GATBackbone(gat_config).to(device)
    node_head = NodeHead(
        input_dim=gat_config.hidden_dim,
        hidden_dim=32
    ).to(device)
    edge_head = EdgeHead(
        input_dim=gat_config.hidden_dim,
        edge_dim=gat_config.edge_input_dim,
        hidden_dim=32
    ).to(device)

    # 4. Controller 생성 (LightPHController)
    controller = LightPHController(
        gat_backbone=gat_backbone,
        node_head=node_head,
        edge_head=edge_head,
        robot_config=robot_config,
        device=device
    )

    # 5. Critic 및 SAC Agent 생성
    state_dim = 4  # [x, y, vx, vy]
    action_dim = 2  # [Fx, Fy]

    critic = CentralizedCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        num_agents=env_config.num_agents
    ).to(device)

    agent = SACAgent(
        actor=controller,
        critic=critic,
        config=training_config,
        device=device
    )

    # 6. Buffer 및 Trainer 생성
    buffer = ReplayBuffer(capacity=training_config.buffer_size)
    trainer = Trainer(
        env=env,
        agent=agent,
        buffer=buffer,
        config=training_config
    )

    # 체크포인트 로드
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # 7. 학습 또는 평가 실행
    if args.mode == 'train':
        # 학습
        history = trainer.train(num_episodes=args.episodes)

        # 최종 모델 저장
        trainer.save_checkpoint(f"{args.save_path}/light_ph_final.pt")

        # 학습 결과 출력
        if history['episode_rewards']:
            print(f"\nTraining Complete!")
            print(f"Final Avg Reward: {np.mean(history['episode_rewards'][-100:]):.2f}")

    else:
        # 평가
        if args.visualize:
            # 시각화 포함 평가
            _run_visual_eval(env, agent, env_config, robot_config)
        else:
            # 통계만 평가
            stats = trainer.evaluate(num_episodes=args.episodes)
            print(f"\nEvaluation Results:")
            print(f"  Avg Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
            print(f"  Avg Length: {stats['avg_length']:.1f}")


def _run_visual_eval(env, agent, env_config, robot_config):
    """시각화 포함 평가 실행."""
    vis = Visualizer(env_config, robot_config)

    obs = env.reset()
    vis.reset()

    running = True
    episode = 0
    step = 0

    def on_key(event):
        nonlocal running, obs, episode, step
        if event.key == 'q':
            running = False
        elif event.key == 'r':
            obs = env.reset()
            vis.trajectories = [[] for _ in range(env_config.num_agents)]
            episode += 1
            step = 0

    vis.fig.canvas.mpl_connect('key_press_event', on_key)

    print("=== Visual Evaluation ===")
    print("Press Q to quit, R to reset")
    print("=========================")

    while running:
        states = obs['states']
        goals = obs['goals']

        # 액션 계산
        actions = []
        for i in range(env_config.num_agents):
            state = states[i]
            goal = goals[i]
            neighbors = [states[j] for j in range(env_config.num_agents) if j != i]

            action = agent.select_action(state, goal, neighbors, explore=False)
            actions.append(action)

        actions = np.array(actions)

        # 환경 스텝
        obs, rewards, done, info = env.step(actions)

        # 시각화
        vis.render(obs['states'], obs['goals'], show_trajectory=True)

        arrived_count = sum(info['arrived'])
        collision_count = len(info['collisions'])
        vis.ax.set_title(
            f'Light-pH | Episode: {episode} | Step: {step} | '
            f'Arrived: {arrived_count}/{env_config.num_agents} | '
            f'Collisions: {collision_count}'
        )

        step += 1

        if done:
            if all(info['arrived']):
                print(f"Episode {episode}: Success in {step} steps!")
            else:
                print(f"Episode {episode}: Ended at step {step}. "
                      f"Arrived: {arrived_count}/{env_config.num_agents}")

            plt.pause(1.0)
            obs = env.reset()
            vis.trajectories = [[] for _ in range(env_config.num_agents)]
            episode += 1
            step = 0

        plt.pause(0.02)

    vis.close()
    print("Evaluation ended.")


if __name__ == "__main__":
    main()
