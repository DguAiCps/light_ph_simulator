"""벽 힘 방향 디버깅 (거리 역비례 공식)."""
import numpy as np

def test_wall_force():
    # 환경: 10x10
    w, h = 10.0, 10.0

    # Agent 위치: (2, 5) - 왼쪽 벽에 가까움
    agent_pos = np.array([2.0, 5.0])

    # 벽 위치 (가장 가까운 점)
    x, y = agent_pos
    wall_positions = [
        np.array([0.0, np.clip(y, 0, h)]),      # Left wall: (0, 5)
        np.array([w, np.clip(y, 0, h)]),        # Right wall: (10, 5)
        np.array([np.clip(x, 0, w), 0.0]),      # Bottom wall: (2, 0)
        np.array([np.clip(x, 0, w), h]),        # Top wall: (2, 10)
    ]

    print(f"Agent 위치: {agent_pos}")
    print(f"환경 크기: {w}x{h}")
    print()
    print("=== 거리 역비례 공식: force = k/r * direction ===")
    print()

    k_io = 1.0  # 스프링 상수

    forces = []
    for i, (name, obs_pos) in enumerate(zip(
        ['Left', 'Right', 'Bottom', 'Top'], wall_positions
    )):
        diff = agent_pos - obs_pos  # q - obs
        dist = np.linalg.norm(diff) + 1e-6
        direction = diff / dist  # 단위 벡터
        force = k_io * direction / dist  # k/r * direction (거리 역비례!)

        forces.append(force)

        print(f"{name} wall:")
        print(f"  벽 위치: {obs_pos}")
        print(f"  거리: {dist:.2f}")
        print(f"  방향 (단위벡터): {direction}")
        print(f"  힘 크기: k/r = {k_io/dist:.4f}")
        print(f"  힘 벡터: {force}")
        print(f"  힘 방향: ", end="")

        if abs(force[0]) > abs(force[1]):
            if force[0] > 0:
                print("오른쪽 → (벽 반대)")
            else:
                print("왼쪽 ← (벽 반대)")
        else:
            if force[1] > 0:
                print("위쪽 ↑ (벽 반대)")
            else:
                print("아래쪽 ↓ (벽 반대)")
        print()

    # 총 힘
    total_force = np.sum(forces, axis=0)
    print(f"=== 총 벽 힘 ===")
    print(f"총 힘: {total_force}")
    print(f"방향: x={total_force[0]:.4f}, y={total_force[1]:.4f}")

    if abs(total_force[0]) > 0.001:
        if total_force[0] > 0:
            print("→ X 방향: 오른쪽 (왼쪽 벽에서 밀림)")
        else:
            print("→ X 방향: 왼쪽 (오른쪽 벽에서 밀림)")

    print()
    print("거리 역비례 덕분에 가까운 벽의 영향이 더 크게 작용합니다!")


if __name__ == "__main__":
    test_wall_force()
