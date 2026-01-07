"""
MAPPO 학습 스크립트.

사용법:
    python train_mappo.py
    python train_mappo.py --resume checkpoints/mappo_final.pt --total_steps 500000
"""

import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, '.')

from config import RobotConfig, EnvConfig
from core import MultiAgentEnv
from training import MAPPO


def main():
    parser = argparse.ArgumentParser(description='MAPPO Training')
    parser.add_argument('--num_agents', type=int, default=4, help='에이전트 수')
    parser.add_argument('--total_steps', type=int, default=100000, help='총 학습 스텝')
    parser.add_argument('--buffer_size', type=int, default=2048, help='버퍼 크기')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Actor 학습률')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Critic 학습률')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='저장 경로')
    parser.add_argument('--resume', type=str, default=None, help='이어서 학습할 체크포인트')
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

    # 환경 생성
    env = MultiAgentEnv(env_config, robot_config)

    print("=== MAPPO Training ===")
    print(f"Agents: {args.num_agents}")
    print(f"Total Steps: {args.total_steps}")
    print(f"Buffer Size: {args.buffer_size}")
    print(f"Batch Size: {args.batch_size}")
    print("======================")

    # MAPPO 생성
    mappo = MAPPO(
        env=env,
        actor_hidden_dim=64,
        critic_hidden_dim=128,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=device
    )

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        mappo.load(args.resume)
        print(f"  Loaded: {mappo.total_steps} steps, {mappo.episode_count} episodes")

    # 학습
    history = mappo.train(
        total_timesteps=args.total_steps,
        log_interval=5,
        save_interval=50,
        save_path=args.save_path
    )

    # 최종 모델 저장
    mappo.save(f"{args.save_path}/mappo_final.pt")
    print(f"\nTraining complete! Model saved to {args.save_path}/mappo_final.pt")

    # 학습 결과 요약
    if history:
        rewards = [h['mean_reward'] for h in history]
        print(f"\nFinal Mean Reward: {rewards[-1]:.2f}")
        print(f"Best Mean Reward: {max(rewards):.2f}")


if __name__ == "__main__":
    main()
