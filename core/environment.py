"""
멀티 에이전트 환경 클래스.

4면 벽으로 둘러싸인 2D 평면에서 에이전트들이 목적지로 이동.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .dynamics import AgentState, Dynamics

try:
    from ..config import EnvConfig, RobotConfig
except ImportError:
    from config import EnvConfig, RobotConfig


class MultiAgentEnv:
    """멀티 에이전트 2D 환경."""

    def __init__(self, env_config: EnvConfig, robot_config: RobotConfig):
        """
        Args:
            env_config: 환경 설정
            robot_config: 로봇 설정
        """
        self.env_config = env_config
        self.robot_config = robot_config

        # 물리 엔진
        self.dynamics = Dynamics(robot_config)

        # 상태 변수
        self.states: List[AgentState] = []
        self.goals: np.ndarray = np.zeros((env_config.num_agents, 2))
        self.step_count: int = 0
        self.prev_distances: np.ndarray = np.zeros(env_config.num_agents)
        self.arrived: np.ndarray = np.zeros(env_config.num_agents, dtype=bool)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        환경 초기화. 에이전트 위치와 목적지 생성.

        Args:
            seed: 랜덤 시드 (선택)

        Returns:
            dict with keys:
                - states: List[AgentState] - 초기 에이전트 상태들
                - goals: np.ndarray shape (N, 2) - 목적지 좌표
        """
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.arrived = np.zeros(self.env_config.num_agents, dtype=bool)

        # 에이전트 초기 위치 생성 (서로 충돌 없이)
        self.states = self._generate_initial_positions()

        # 교차하는 목적지 생성
        self.goals = self._generate_crossing_goals()

        # 초기 거리 저장 (보상 계산용)
        self.prev_distances = np.array([
            np.linalg.norm(self.states[i].position - self.goals[i])
            for i in range(self.env_config.num_agents)
        ])

        return {
            'states': self.states,
            'goals': self.goals.copy()
        }

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, bool, Dict[str, Any]]:
        """
        한 스텝 진행.

        Args:
            actions: shape (N, 2) - 각 에이전트의 힘 입력

        Returns:
            tuple of:
                - obs: dict - 새로운 observation
                - rewards: np.ndarray shape (N,) - 각 에이전트 보상
                - done: bool - 에피소드 종료 여부
                - info: dict - 추가 정보 (충돌 여부, 도착 여부 등)
        """
        actions = np.asarray(actions, dtype=np.float64)
        assert actions.shape == (self.env_config.num_agents, 2)

        self.step_count += 1
        collisions = []
        rewards = np.zeros(self.env_config.num_agents)

        # 각 에이전트에 대해 물리 시뮬레이션
        new_states = []
        for i in range(self.env_config.num_agents):
            # Dynamics 적용
            new_state = self.dynamics.step(
                self.states[i],
                actions[i],
                self.env_config.dt
            )

            # Hard constraint: 벽 충돌 처리 (현실적 - 튕김 없음)
            new_state = self._apply_wall_constraint(new_state)

            new_states.append(new_state)

        self.states = new_states

        # 에이전트 간 충돌 검사 (물리 제약 적용 전에 검사해야 함!)
        collisions = self._check_collision()

        # 에이전트 간 충돌 제약 (Hard constraint - 벽처럼)
        self._apply_agent_collision_constraint()

        # 보상 계산
        for i in range(self.env_config.num_agents):
            rewards[i] = self._compute_reward(i, collisions)

        # 도착 여부 업데이트
        for i in range(self.env_config.num_agents):
            dist = np.linalg.norm(self.states[i].position - self.goals[i])
            if dist < self.env_config.goal_threshold:
                self.arrived[i] = True

        # 현재 거리 저장 (다음 스텝 보상 계산용)
        self.prev_distances = np.array([
            np.linalg.norm(self.states[i].position - self.goals[i])
            for i in range(self.env_config.num_agents)
        ])

        # 종료 조건 확인
        done = self._is_done()

        obs = {
            'states': self.states,
            'goals': self.goals.copy()
        }

        info = {
            'collisions': collisions,
            'arrived': self.arrived.copy(),
            'step_count': self.step_count
        }

        return obs, rewards, done, info

    def _generate_initial_positions(self) -> List[AgentState]:
        """
        에이전트 초기 위치 생성 (서로 충돌 없이).

        Returns:
            초기 에이전트 상태 리스트
        """
        states = []
        min_distance = 2.5 * self.robot_config.radius  # 최소 거리
        margin = self.robot_config.radius + 0.1  # 벽으로부터 여유

        for _ in range(self.env_config.num_agents):
            max_attempts = 100
            for attempt in range(max_attempts):
                # 랜덤 위치 생성
                x = np.random.uniform(margin, self.env_config.width - margin)
                y = np.random.uniform(margin, self.env_config.height - margin)
                position = np.array([x, y])

                # 기존 에이전트와 충돌 검사
                valid = True
                for existing in states:
                    dist = np.linalg.norm(position - existing.position)
                    if dist < min_distance:
                        valid = False
                        break

                if valid:
                    states.append(AgentState(position=position, velocity=np.zeros(2)))
                    break

            if len(states) < _ + 1:
                # 실패 시 강제 배치
                x = margin + (_ % 2) * (self.env_config.width - 2 * margin)
                y = margin + (_ // 2) * (self.env_config.height - 2 * margin)
                states.append(AgentState(position=np.array([x, y]), velocity=np.zeros(2)))

        return states

    def _generate_crossing_goals(self) -> np.ndarray:
        """
        에이전트들이 교차하도록 목적지 생성.
        대각선 스왑 방식: 에이전트를 4분면에 배치 후 대각선 반대편을 목적지로.

        Returns:
            목적지 좌표 (N, 2)
        """
        goals = np.zeros((self.env_config.num_agents, 2))
        margin = self.robot_config.radius + 0.2

        if self.env_config.num_agents == 4:
            # 4분면 대각선 교차
            for i in range(4):
                # 현재 에이전트 위치의 대각선 반대편
                current_pos = self.states[i].position
                center = np.array([self.env_config.width / 2, self.env_config.height / 2])

                # 중심 기준 반대편으로 목적지 설정
                offset = current_pos - center
                goal = center - offset

                # 범위 내로 클리핑
                goal[0] = np.clip(goal[0], margin, self.env_config.width - margin)
                goal[1] = np.clip(goal[1], margin, self.env_config.height - margin)

                goals[i] = goal
        else:
            # 일반적인 경우: 랜덤 목적지 (교차 유도)
            min_goal_dist = 2.5 * self.robot_config.radius  # 목표 간 최소 거리

            for i in range(self.env_config.num_agents):
                max_attempts = 100
                for attempt in range(max_attempts):
                    x = np.random.uniform(margin, self.env_config.width - margin)
                    y = np.random.uniform(margin, self.env_config.height - margin)
                    goal = np.array([x, y])

                    # 자기 위치와 충분히 멀어야 함
                    if np.linalg.norm(goal - self.states[i].position) < self.env_config.width / 4:
                        continue

                    # 기존 목표들과 겹치지 않아야 함
                    valid = True
                    for j in range(i):
                        if np.linalg.norm(goal - goals[j]) < min_goal_dist:
                            valid = False
                            break

                    if valid:
                        goals[i] = goal
                        break

                # 실패 시 그냥 배치
                if attempt == max_attempts - 1:
                    goals[i] = goal

        return goals

    def _check_collision(self) -> List[Tuple[int, int]]:
        """
        에이전트 간 근접 검사 (페널티용).

        물리적 충돌(겹침)은 _apply_agent_collision_constraint에서 처리.
        여기서는 근접 시 페널티를 주기 위한 검사.

        Returns:
            근접한 에이전트 쌍의 인덱스 리스트
        """
        collisions = []
        # 물리적 접촉 시에만 페널티 (마진 없음)
        # 물리 충돌 제약으로 겹침은 방지되지만, 접촉 시도 자체에 페널티
        proximity_dist = 2 * self.robot_config.radius

        for i in range(self.env_config.num_agents):
            for j in range(i + 1, self.env_config.num_agents):
                dist = np.linalg.norm(
                    self.states[i].position - self.states[j].position
                )
                if dist < proximity_dist:
                    collisions.append((i, j))

        return collisions

    def _apply_wall_constraint(self, state: AgentState) -> AgentState:
        """
        벽 충돌 시 위치 클리핑 및 속도 제거 (Hard constraint).

        Args:
            state: 에이전트 상태

        Returns:
            수정된 에이전트 상태
        """
        pos = state.position.copy()
        vel = state.velocity.copy()
        radius = self.robot_config.radius

        # 왼쪽 벽
        if pos[0] < radius:
            pos[0] = radius
            vel[0] = max(vel[0], 0)  # 벽 방향 속도만 제거

        # 오른쪽 벽
        if pos[0] > self.env_config.width - radius:
            pos[0] = self.env_config.width - radius
            vel[0] = min(vel[0], 0)

        # 아래쪽 벽
        if pos[1] < radius:
            pos[1] = radius
            vel[1] = max(vel[1], 0)

        # 위쪽 벽
        if pos[1] > self.env_config.height - radius:
            pos[1] = self.env_config.height - radius
            vel[1] = min(vel[1], 0)

        return AgentState(position=pos, velocity=vel)

    def _apply_agent_collision_constraint(self):
        """
        에이전트 간 충돌 시 위치 분리 및 속도 조정 (Hard constraint).

        벽처럼 서로 통과하지 못하도록 함.
        여러 에이전트가 동시에 겹칠 수 있으므로 반복 처리.
        """
        radius = self.robot_config.radius
        min_dist = 2 * radius  # 두 에이전트 중심 간 최소 거리

        # 여러 번 반복하여 모든 충돌 해결 (최대 10회)
        for _ in range(10):
            resolved = True

            for i in range(self.env_config.num_agents):
                for j in range(i + 1, self.env_config.num_agents):
                    pos_i = self.states[i].position
                    pos_j = self.states[j].position
                    vel_i = self.states[i].velocity
                    vel_j = self.states[j].velocity

                    # 두 에이전트 간 거리
                    diff = pos_j - pos_i
                    dist = np.linalg.norm(diff)

                    if dist < min_dist and dist > 1e-6:
                        resolved = False

                        # 겹침량
                        overlap = min_dist - dist

                        # 분리 방향 (i에서 j로)
                        direction = diff / dist

                        # 각각 절반씩 밀어내기
                        separation = direction * (overlap / 2 + 1e-4)
                        new_pos_i = pos_i - separation
                        new_pos_j = pos_j + separation

                        # 속도 조정: 서로를 향하는 속도 성분 제거
                        vel_along_i = np.dot(vel_i, direction)
                        vel_along_j = np.dot(vel_j, -direction)

                        new_vel_i = vel_i.copy()
                        new_vel_j = vel_j.copy()

                        # i가 j 방향으로 움직이면 그 성분 제거
                        if vel_along_i > 0:
                            new_vel_i = vel_i - vel_along_i * direction

                        # j가 i 방향으로 움직이면 그 성분 제거
                        if vel_along_j > 0:
                            new_vel_j = vel_j - vel_along_j * (-direction)

                        # 상태 업데이트
                        self.states[i] = AgentState(position=new_pos_i, velocity=new_vel_i)
                        self.states[j] = AgentState(position=new_pos_j, velocity=new_vel_j)

                    elif dist <= 1e-6:
                        # 완전히 겹친 경우: 랜덤 방향으로 분리
                        resolved = False
                        direction = np.random.randn(2)
                        direction = direction / (np.linalg.norm(direction) + 1e-8)

                        separation = direction * (min_dist / 2 + 1e-4)
                        new_pos_i = pos_i - separation
                        new_pos_j = pos_j + separation

                        self.states[i] = AgentState(position=new_pos_i, velocity=np.zeros(2))
                        self.states[j] = AgentState(position=new_pos_j, velocity=np.zeros(2))

            # 분리 후 벽 제약 다시 적용
            for i in range(self.env_config.num_agents):
                self.states[i] = self._apply_wall_constraint(self.states[i])

            if resolved:
                break

    def _compute_reward(self, agent_idx: int, collisions: List[Tuple[int, int]]) -> float:
        """
        단일 에이전트의 보상 계산.
        r_t = η₁ Δμᵢ - η₂ Σⱼ 𝟙[rᵢⱼ < r_safe] + η₃ 𝟙[μᵢ = 1]
        Args:
            agent_idx: 에이전트 인덱스
            collisions: 충돌 쌍 리스트

        Returns:
            보상 값
        """
        reward = 0.0
        pos = self.states[agent_idx].position
        current_dist = np.linalg.norm(pos - self.goals[agent_idx])

        # 1. 에이전트 충돌 페널티
        for (i, j) in collisions:
            if agent_idx == i or agent_idx == j:
                reward += self.env_config.collision_penalty  # -10.0
                break

        # 1b. 근접 페널티 (충돌 전에도 가까우면 페널티)
        proximity_threshold = 3.0 * self.robot_config.radius  # ~0.315m
        collision_dist = 2.0 * self.robot_config.radius  # ~0.21m
        for i in range(self.env_config.num_agents):
            if i != agent_idx:
                dist = np.linalg.norm(pos - self.states[i].position)
                if dist < proximity_threshold and dist > collision_dist:
                    # 가까울수록 페널티 증가 (최대 -0.5)
                    proximity_ratio = 1.0 - (dist - collision_dist) / (proximity_threshold - collision_dist)
                    reward -= 0.5 * proximity_ratio

        # 2. 벽 충돌 페널티
        wall_contact_dist = self.robot_config.radius + 0.01
        if (pos[0] < wall_contact_dist or
            pos[0] > self.env_config.width - wall_contact_dist or
            pos[1] < wall_contact_dist or
            pos[1] > self.env_config.height - wall_contact_dist):
            reward -= 5.0

        # 3. 도착 보너스 (최초 도착 시 1회)
        if current_dist < self.env_config.goal_threshold and not self.arrived[agent_idx]:
            reward += self.env_config.goal_reward  # +10.0

        # 4. Goal 유지 보상 (도착 후 매 스텝)
        if current_dist < self.env_config.goal_threshold:
            reward += 1.0  # 매 스텝 +1.0

        # 5. 도착 후 이탈 페널티
        if self.arrived[agent_idx]:
            if current_dist >= self.env_config.goal_threshold:
                reward -= 5.0

        return reward

    def _is_done(self) -> bool:
        """
        에피소드 종료 조건 확인.

        Returns:
            종료 여부
        """
        # 모든 에이전트 도착
        if np.all(self.arrived):
            return True

        # max_steps 도달
        if self.step_count >= self.env_config.max_steps:
            return True

        return False

    def get_observation(self, agent_idx: int) -> np.ndarray:
        """
        단일 에이전트 관점의 observation 생성.

        Args:
            agent_idx: 에이전트 인덱스

        Returns:
            observation 벡터 [자기 상태, 목표 오프셋, 이웃 상태들]
        """
        state = self.states[agent_idx]
        goal = self.goals[agent_idx]

        # 자기 상태: [x, y, vx, vy]
        self_state = np.concatenate([state.position, state.velocity])

        # 목표 오프셋: [dx, dy]
        goal_offset = goal - state.position

        # 이웃 상태들: [(x, y, vx, vy), ...]
        neighbor_states = []
        for i in range(self.env_config.num_agents):
            if i != agent_idx:
                neighbor = self.states[i]
                # 상대적 위치와 속도
                rel_pos = neighbor.position - state.position
                rel_vel = neighbor.velocity - state.velocity
                neighbor_states.extend([*rel_pos, *rel_vel])

        return np.concatenate([self_state, goal_offset, neighbor_states])
