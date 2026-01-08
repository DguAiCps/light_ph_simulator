"""
Light-pH 학습 스크립트.

사용법:
    python train_light_ph.py
    python train_light_ph.py --resume checkpoints/light_ph_final.pt --num_episodes 1000
"""

import sys
import argparse
import torch

sys.path.insert(0, '.')

from config import RobotConfig, EnvConfig, GATConfig, TrainingConfig
from core import MultiAgentEnv
from networks import GATBackbone, NodeHead, EdgeHead
from controller import LightPHController
from training import CentralizedCritic, SACAgent, ReplayBuffer, Trainer


def main():
    parser = argparse.ArgumentParser(description='Light-pH Training')
    parser.add_argument('--num_agents', type=int, default=4, help='에이전트 수')
    parser.add_argument('--num_episodes', type=int, default=500, help='학습 에피소드 수')
    parser.add_argument('--buffer_size', type=int, default=100000, help='버퍼 크기')
    parser.add_argument('--batch_size', type=int, default=256, help='배치 크기')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup 스텝')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Actor 학습률')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='Critic 학습률')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='저장 경로')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='체크포인트 저장 주기 (에피소드)')
    parser.add_argument('--resume', type=str, default=None, help='이어서 학습할 체크포인트')
    parser.add_argument('--device', type=str, default='cuda', help='디바이스')
    parser.add_argument('--no_walls', action='store_true', help='벽을 장애물로 그래프에서 제외 (기본: 포함)')
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

    # 환경 생성
    env = MultiAgentEnv(env_config, robot_config)

    print("=== Light-pH Training ===")
    print(f"Agents: {args.num_agents}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Buffer Size: {args.buffer_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Use Walls: {not args.no_walls}")
    print(f"Checkpoint: every {args.checkpoint_interval} episodes")
    print("=========================")

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
        device=device,
        env_config=env_config if not args.no_walls else None
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

    # Trainer
    trainer = Trainer(
        env=env,
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
        num_episodes=args.num_episodes,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=args.save_path
    )

    # 최종 모델 저장
    trainer.save_checkpoint(f"{args.save_path}/light_ph_final.pt")
    print(f"\nTraining complete! Model saved to {args.save_path}/light_ph_final.pt")

    # 평가
    print("\n=== Evaluation ===")
    eval_result = trainer.evaluate(num_episodes=20)
    print(f"Success Rate: {eval_result['success_rate']:.2%}")
    print(f"Avg Reward: {eval_result['avg_reward']:.2f}")
    print(f"Avg Length: {eval_result['avg_length']:.1f}")


if __name__ == "__main__":
    main()
