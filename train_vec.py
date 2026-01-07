"""
Light-pH Vectorized 학습 스크립트.

여러 환경을 동시에 실행하여 데이터 수집 속도 향상.

사용법:
    python train_vec.py
    python train_vec.py --num_envs 16 --total_steps 500000
    python train_vec.py --resume checkpoints/light_ph_final.pt
"""

import sys
import argparse
import torch

sys.path.insert(0, '.')

from config import RobotConfig, EnvConfig, GATConfig, TrainingConfig
from core import VectorizedEnv
from networks import GATBackbone, NodeHead, EdgeHead
from controller import LightPHController
from training import CentralizedCritic, SACAgent, ReplayBuffer, VectorizedTrainer


def main():
    parser = argparse.ArgumentParser(description='Light-pH Vectorized Training')
    parser.add_argument('--num_agents', type=int, default=4, help='에이전트 수')
    parser.add_argument('--num_envs', type=int, default=8, help='병렬 환경 수')
    parser.add_argument('--total_steps', type=int, default=300000, help='총 학습 스텝 수')
    parser.add_argument('--buffer_size', type=int, default=100000, help='버퍼 크기')
    parser.add_argument('--batch_size', type=int, default=256, help='배치 크기')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup 스텝')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Actor 학습률')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='Critic 학습률')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='저장 경로')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='체크포인트 저장 주기 (스텝)')
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
    gat_config = GATConfig()
    training_config = TrainingConfig(
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        warmup_steps=args.warmup_steps
    )

    # Vectorized 환경 생성
    vec_env = VectorizedEnv(env_config, robot_config, num_envs=args.num_envs)

    print("=== Light-pH Vectorized Training ===")
    print(f"Agents: {args.num_agents}")
    print(f"Parallel Envs: {args.num_envs}")
    print(f"Total Steps: {args.total_steps}")
    print(f"Buffer Size: {args.buffer_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Checkpoint: every {args.checkpoint_interval} steps")
    print("=====================================")

    # GAT 네트워크 생성
    gat_backbone = GATBackbone(config=gat_config)

    node_head = NodeHead(
        input_dim=gat_config.hidden_dim,
        hidden_dim=32
    )

    edge_head = EdgeHead(
        input_dim=gat_config.hidden_dim,
        edge_dim=gat_config.edge_input_dim,
        hidden_dim=32
    )

    # Light-pH Controller (Actor)
    actor = LightPHController(
        gat_backbone=gat_backbone,
        node_head=node_head,
        edge_head=edge_head,
        robot_config=robot_config,
        device=device
    )

    # Centralized Critic
    state_dim = 4  # [x, y, vx, vy]
    action_dim = 2  # [Fx, Fy]
    critic = CentralizedCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        num_agents=args.num_agents
    )
    critic.to(device)

    # SAC Agent
    agent = SACAgent(
        actor=actor,
        critic=critic,
        config=training_config,
        device=device
    )

    # Replay Buffer
    buffer = ReplayBuffer(capacity=args.buffer_size)

    # Vectorized Trainer
    trainer = VectorizedTrainer(
        vec_env=vec_env,
        agent=agent,
        buffer=buffer,
        config=training_config
    )

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 학습
    history = trainer.train(
        total_steps=args.total_steps,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=args.save_path
    )

    # 최종 모델 저장
    trainer.save_checkpoint(f"{args.save_path}/light_ph_vec_final.pt")
    print(f"\nTraining complete! Model saved to {args.save_path}/light_ph_vec_final.pt")

    # 결과 요약
    print("\n=== Training Summary ===")
    print(f"Total Episodes: {trainer.episode_count}")
    print(f"Total Steps: {trainer.total_steps}")
    if history['episode_rewards']:
        print(f"Final Avg Reward: {sum(history['episode_rewards'][-100:]) / min(100, len(history['episode_rewards'])):.2f}")


if __name__ == "__main__":
    main()
