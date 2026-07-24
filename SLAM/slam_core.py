"""Pure NumPy dynamics for cooperative partially observable SLAM.

The core contains no Gymnasium, MiniGrid, TensorFlow, Cirq, or TFQ dependency.
Map generation, synchronous transitions, local mapping, reward, and metrics can
therefore be tested independently of the learning stack.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import atan2, hypot, pi
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Coordinate = Tuple[int, int]  # (x, y); arrays always use [y, x].

UNKNOWN = np.uint8(0)
FREE = np.uint8(1)
WALL = np.uint8(2)

ACTION_FORWARD = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
N_ACTIONS = 3

OBSERVATION_CHANNELS = (
    "unknown",
    "known_free",
    "known_wall",
    "current_fov",
    "visited_by_self",
)
OBS_CHANNELS = len(OBSERVATION_CHANNELS)
AUX_DIM = 6  # normalized x/y + four-way heading one-hot

_DIRECTION_VECTORS: Tuple[Coordinate, ...] = (
    (1, 0),   # east
    (0, 1),   # south
    (-1, 0),  # west
    (0, -1),  # north
)


@dataclass(frozen=True)
class RewardConfig:
    """Scale-free cooperative reward configuration."""

    information_gain_scale: float = 10.0
    step_penalty: float = 0.01
    collision_penalty: float = 0.05
    redundancy_penalty: float = 0.0
    milestone_25_bonus: float = 0.10
    milestone_50_bonus: float = 0.20
    milestone_75_bonus: float = 0.40
    completion_bonus: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class MovementResult:
    positions: Tuple[Coordinate, ...]
    directions: Tuple[int, ...]
    collisions: Tuple[bool, ...]
    moved: Tuple[bool, ...]
    turned: Tuple[bool, ...]
    forward_attempts: Tuple[bool, ...]


class CooperativeSLAMCore:
    """Synchronous multi-agent exploration with strictly local belief maps.

    Ground-truth walls are used by the simulator only.  An agent's observation
    contains a cell only after that cell enters the agent's own line of sight.
    The team union map is used for reward, termination, and evaluation; it is
    never inserted directly into a decentralized actor observation.
    """

    def __init__(
        self,
        size: int = 8,
        n_agents: int = 2,
        max_steps: int = 100,
        maze_type: str = "random",
        obstacle_density: float = 0.15,
        fov_range: int = 4,
        fov_angle: float = pi / 2.0,
        target_coverage: float = 0.95,
        reward_config: Optional[RewardConfig] = None,
        agent_start_positions: Optional[Sequence[Coordinate]] = None,
        agent_start_dirs: Optional[Sequence[int]] = None,
    ) -> None:
        if isinstance(size, bool) or int(size) != size or size < 5:
            raise ValueError("size must be an integer of at least 5")
        if isinstance(n_agents, bool) or int(n_agents) != n_agents or n_agents < 1:
            raise ValueError("n_agents must be a positive integer")
        if isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if int(n_agents) > (int(size) - 2) ** 2:
            raise ValueError("n_agents exceeds the number of interior cells")
        maze_type = str(maze_type).strip().lower()
        if maze_type not in {"empty", "hard", "random"}:
            raise ValueError("maze_type must be empty, hard, or random")
        if not np.isfinite(obstacle_density) or not 0.0 <= obstacle_density < 0.5:
            raise ValueError("obstacle_density must be finite and in [0, 0.5)")
        if not np.isfinite(target_coverage) or not 0.0 < target_coverage <= 1.0:
            raise ValueError("target_coverage must be finite and in (0, 1]")
        if isinstance(fov_range, bool) or int(fov_range) != fov_range or fov_range < 1:
            raise ValueError("fov_range must be a positive integer")
        if not np.isfinite(fov_angle) or fov_angle <= 0.0 or fov_angle > 2.0 * pi:
            raise ValueError("fov_angle must be finite and in (0, 2*pi]")
        reward_config = reward_config or RewardConfig()
        for field_name, field_value in reward_config.to_dict().items():
            if not np.isfinite(field_value) or field_value < 0.0:
                raise ValueError(
                    f"reward coefficient {field_name} must be non-negative and finite"
                )

        self.size = int(size)
        self.n_agents = int(n_agents)
        self.max_steps = int(max_steps)
        self.maze_type = maze_type
        self.obstacle_density = float(obstacle_density)
        self.fov_range = int(max(1, min(int(fov_range), self.size - 2)))
        self.fov_angle = float(fov_angle)
        self.target_coverage = float(target_coverage)
        self.reward_config = reward_config

        if agent_start_positions is not None:
            validated_positions: List[Coordinate] = []
            for point in agent_start_positions:
                if len(point) != 2:
                    raise ValueError("each agent start position must be an (x, y) pair")
                x, y = point
                if (
                    isinstance(x, bool)
                    or isinstance(y, bool)
                    or int(x) != x
                    or int(y) != y
                ):
                    raise ValueError("agent start coordinates must be integers")
                validated_positions.append((int(x), int(y)))
            self._requested_start_positions = tuple(validated_positions)
        else:
            self._requested_start_positions = None

        if agent_start_dirs is not None:
            validated_directions: List[int] = []
            for direction in agent_start_dirs:
                if isinstance(direction, bool) or int(direction) != direction:
                    raise ValueError("agent start directions must be integers")
                validated_directions.append(int(direction) % 4)
            self._requested_start_dirs = tuple(validated_directions)
        else:
            self._requested_start_dirs = None
        if self._requested_start_positions is not None:
            if len(set(self._requested_start_positions)) != len(
                self._requested_start_positions
            ):
                raise ValueError("agent_start_positions must be unique")
            for x, y in self._requested_start_positions:
                if not (1 <= x < int(size) - 1 and 1 <= y < int(size) - 1):
                    raise ValueError(
                        "agent_start_positions must lie in interior grid cells"
                    )
        if (
            self._requested_start_positions is not None
            and len(self._requested_start_positions) != self.n_agents
        ):
            raise ValueError("agent_start_positions length must equal n_agents")
        if (
            self._requested_start_dirs is not None
            and len(self._requested_start_dirs) != self.n_agents
        ):
            raise ValueError("agent_start_dirs length must equal n_agents")

        self.rng = np.random.default_rng()
        self.seed_value: Optional[int] = None
        self._fov_offsets = self._build_fov_offsets()

        self.wall_map = np.zeros((self.size, self.size), dtype=bool)
        self.observable_mask = np.zeros_like(self.wall_map)
        self.known_maps = np.zeros(
            (self.n_agents, self.size, self.size), dtype=np.uint8
        )
        self.current_fov_maps = np.zeros_like(self.known_maps, dtype=bool)
        self.visited_maps = np.zeros_like(self.known_maps, dtype=bool)
        self.team_explored_map = np.zeros_like(self.wall_map, dtype=bool)
        self.agent_positions: List[Coordinate] = []
        self.agent_dirs: List[int] = []
        self.step_count = 0
        self.collisions_total = 0
        self.turns_total = 0
        self.forward_attempts_total = 0
        self.moves_total = 0
        self._milestones_reached: Set[float] = set()
        self._last_metrics: Dict[str, object] = {}
        self._episode_done = False

    # ------------------------------------------------------------------
    # Reset and map generation
    # ------------------------------------------------------------------
    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)

        requested_positions = self._preferred_start_positions()
        self.wall_map = self._generate_wall_map(requested_positions)
        self.agent_positions = self._select_start_positions(requested_positions)
        self.agent_dirs = list(self._preferred_start_dirs())
        self.observable_mask = self._compute_observable_mask(self.agent_positions[0])

        self.known_maps.fill(UNKNOWN)
        self.current_fov_maps.fill(False)
        self.visited_maps.fill(False)
        self.team_explored_map.fill(False)
        for index, (x, y) in enumerate(self.agent_positions):
            self.visited_maps[index, y, x] = True

        self.step_count = 0
        self.collisions_total = 0
        self.turns_total = 0
        self.forward_attempts_total = 0
        self.moves_total = 0

        fov_sets = [self._compute_fov(index) for index in range(self.n_agents)]
        self._apply_fov_updates(fov_sets)
        self._milestones_reached = {
            threshold
            for threshold in (0.25, 0.50, 0.75)
            if self.coverage >= threshold
        }
        success = self.coverage >= self.target_coverage
        self._episode_done = bool(success)
        zeros = np.zeros(self.n_agents, dtype=bool)
        self._last_metrics = self._build_metrics(
            new_cells=0,
            collisions=zeros,
            moved=zeros,
            turned=zeros,
            forward_attempts=zeros,
            overlap_cells=self._fov_overlap_cells(fov_sets),
            team_reward=0.0,
            terminated=success,
            truncated=False,
        )
        return self.observation(), {
            "seed": self.seed_value,
            "metrics": dict(self._last_metrics),
            "reward_config": self.reward_config.to_dict(),
        }

    def _preferred_start_positions(self) -> Tuple[Coordinate, ...]:
        if self._requested_start_positions is not None:
            return self._requested_start_positions
        candidates: List[Coordinate] = [
            (1, 1),
            (self.size - 2, self.size - 2),
            (self.size - 2, 1),
            (1, self.size - 2),
            (self.size // 2, 1),
            (self.size // 2, self.size - 2),
            (1, self.size // 2),
            (self.size - 2, self.size // 2),
        ]
        interior = [
            (x, y)
            for y in range(1, self.size - 1)
            for x in range(1, self.size - 1)
            if (x, y) not in candidates
        ]
        candidates.extend(interior)
        if self.n_agents > len(candidates):
            raise ValueError("n_agents exceeds the number of interior cells")
        return tuple(candidates[: self.n_agents])

    def _preferred_start_dirs(self) -> Tuple[int, ...]:
        if self._requested_start_dirs is not None:
            return self._requested_start_dirs
        base = (0, 2, 1, 3)
        return tuple(base[index % len(base)] for index in range(self.n_agents))

    def _generate_wall_map(
        self, reserved_positions: Sequence[Coordinate]
    ) -> np.ndarray:
        walls = np.zeros((self.size, self.size), dtype=bool)
        walls[0, :] = True
        walls[-1, :] = True
        walls[:, 0] = True
        walls[:, -1] = True

        if self.maze_type == "hard":
            self._add_hard_maze(walls, reserved_positions)
        elif self.maze_type == "random":
            self._add_random_obstacles(walls, reserved_positions)

        if not self._free_cells_connected(walls):
            raise RuntimeError("map generator produced a disconnected free-space graph")
        return walls

    def _add_hard_maze(
        self, walls: np.ndarray, reserved_positions: Sequence[Coordinate]
    ) -> None:
        center_x = self.size // 2
        center_y = self.size // 2
        walls[center_y, 1:-1] = True
        walls[1:-1, center_x] = True

        gaps = {
            (max(1, self.size // 4), center_y),
            (min(self.size - 2, (3 * self.size) // 4), center_y),
            (center_x, max(1, self.size // 4)),
            (center_x, min(self.size - 2, (3 * self.size) // 4)),
        }
        for x, y in gaps:
            walls[y, x] = False
        for x, y in reserved_positions:
            if 0 < x < self.size - 1 and 0 < y < self.size - 1:
                walls[y, x] = False

        # Very small grids can merge gaps.  Restore connectivity defensively.
        if not self._free_cells_connected(walls):
            cross_cells = [
                (x, center_y) for x in range(1, self.size - 1)
            ] + [
                (center_x, y) for y in range(1, self.size - 1)
            ]
            cross_cells.sort(
                key=lambda point: abs(point[0] - center_x) + abs(point[1] - center_y)
            )
            for x, y in cross_cells:
                walls[y, x] = False
                if self._free_cells_connected(walls):
                    break

    def _add_random_obstacles(
        self, walls: np.ndarray, reserved_positions: Sequence[Coordinate]
    ) -> None:
        reserved: Set[Coordinate] = set()
        for x, y in reserved_positions:
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 < nx < self.size - 1 and 0 < ny < self.size - 1:
                    reserved.add((nx, ny))

        candidates = [
            (x, y)
            for y in range(1, self.size - 1)
            for x in range(1, self.size - 1)
            if (x, y) not in reserved
        ]
        self.rng.shuffle(candidates)
        target = min(
            int(round((self.size - 2) ** 2 * self.obstacle_density)),
            len(candidates),
        )
        placed = 0
        for x, y in candidates:
            if placed >= target:
                break
            walls[y, x] = True
            if self._free_cells_connected(walls):
                placed += 1
            else:
                walls[y, x] = False

    def _select_start_positions(
        self, preferred: Sequence[Coordinate]
    ) -> List[Coordinate]:
        free_positions = [
            (x, y)
            for y in range(1, self.size - 1)
            for x in range(1, self.size - 1)
            if not self.wall_map[y, x]
        ]
        selected: List[Coordinate] = []
        used: Set[Coordinate] = set()
        for requested_x, requested_y in preferred:
            available = [point for point in free_positions if point not in used]
            if not available:
                raise RuntimeError("not enough free cells for all agents")
            available.sort(
                key=lambda point: (
                    abs(point[0] - requested_x) + abs(point[1] - requested_y),
                    point[1],
                    point[0],
                )
            )
            selected.append(available[0])
            used.add(available[0])
        return selected

    @staticmethod
    def _neighbors4(x: int, y: int) -> Iterable[Coordinate]:
        for dx, dy in _DIRECTION_VECTORS:
            yield x + dx, y + dy

    def _free_cells_connected(self, walls: np.ndarray) -> bool:
        free = ~walls
        coordinates = np.argwhere(free)
        if coordinates.size == 0:
            return False
        start_y, start_x = map(int, coordinates[0])
        visited = np.zeros_like(free, dtype=bool)
        visited[start_y, start_x] = True
        queue: deque[Coordinate] = deque([(start_x, start_y)])
        while queue:
            x, y = queue.popleft()
            for nx, ny in self._neighbors4(x, y):
                if (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and free[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        return bool(np.all(visited[free]))

    def _reachable_free_mask(self, start: Coordinate) -> np.ndarray:
        reachable = np.zeros_like(self.wall_map, dtype=bool)
        start_x, start_y = start
        if self.wall_map[start_y, start_x]:
            return reachable
        reachable[start_y, start_x] = True
        queue: deque[Coordinate] = deque([(start_x, start_y)])
        while queue:
            x, y = queue.popleft()
            for nx, ny in self._neighbors4(x, y):
                if (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and not self.wall_map[ny, nx]
                    and not reachable[ny, nx]
                ):
                    reachable[ny, nx] = True
                    queue.append((nx, ny))
        return reachable

    def _compute_observable_mask(self, start: Coordinate) -> np.ndarray:
        """Reachable free cells plus boundary walls that can be observed."""
        reachable = self._reachable_free_mask(start)
        observable = reachable.copy()
        ys, xs = np.where(reachable)
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < self.size
                        and 0 <= ny < self.size
                        and self.wall_map[ny, nx]
                    ):
                        observable[ny, nx] = True
        return observable

    # ------------------------------------------------------------------
    # Field of view and local mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + pi) % (2.0 * pi) - pi

    def _build_fov_offsets(self) -> Dict[int, Tuple[Coordinate, ...]]:
        offsets_by_direction: Dict[int, Tuple[Coordinate, ...]] = {}
        for direction in range(4):
            heading = direction * pi / 2.0
            offsets: List[Coordinate] = [(0, 0)]
            for dy in range(-self.fov_range, self.fov_range + 1):
                for dx in range(-self.fov_range, self.fov_range + 1):
                    if dx == 0 and dy == 0:
                        continue
                    distance = hypot(dx, dy)
                    if distance > self.fov_range + 1e-9:
                        continue
                    angle = atan2(dy, dx)
                    if (
                        abs(self._wrap_angle(angle - heading))
                        <= self.fov_angle / 2.0 + 1e-9
                    ):
                        offsets.append((dx, dy))
            offsets.sort(key=lambda point: (hypot(*point), point[1], point[0]))
            offsets_by_direction[direction] = tuple(offsets)
        return offsets_by_direction

    @staticmethod
    def _bresenham_line(
        x0: int, y0: int, x1: int, y1: int
    ) -> List[Coordinate]:
        points: List[Coordinate] = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy
        return points

    def _line_of_sight(self, origin: Coordinate, target: Coordinate) -> bool:
        # A target wall is visible; only intermediate walls occlude it.
        for x, y in self._bresenham_line(*origin, *target)[1:-1]:
            if self.wall_map[y, x]:
                return False
        return True

    def _compute_fov(self, agent_index: int) -> Set[Coordinate]:
        agent_x, agent_y = self.agent_positions[agent_index]
        direction = self.agent_dirs[agent_index]
        visible: Set[Coordinate] = set()
        for dx, dy in self._fov_offsets[direction]:
            target_x, target_y = agent_x + dx, agent_y + dy
            if not (0 <= target_x < self.size and 0 <= target_y < self.size):
                continue
            if self._line_of_sight(
                (agent_x, agent_y), (target_x, target_y)
            ):
                visible.add((target_x, target_y))
        visible.add((agent_x, agent_y))
        return visible

    def _apply_fov_updates(
        self, fov_sets: Sequence[Set[Coordinate]]
    ) -> Tuple[int, np.ndarray]:
        previous_team_map = self.team_explored_map.copy()
        previous_known = self.known_maps.copy()
        self.current_fov_maps.fill(False)
        per_agent_new = np.zeros(self.n_agents, dtype=np.int64)
        for agent_index, visible in enumerate(fov_sets):
            for x, y in visible:
                self.current_fov_maps[agent_index, y, x] = True
                self.known_maps[agent_index, y, x] = (
                    WALL if self.wall_map[y, x] else FREE
                )
            newly_free = (
                (self.known_maps[agent_index] == FREE)
                & (previous_known[agent_index] != FREE)
                & self.observable_mask
            )
            per_agent_new[agent_index] = int(np.count_nonzero(newly_free))
        self.team_explored_map = np.any(self.known_maps != UNKNOWN, axis=0)
        newly_explored = (
            self.team_explored_map & self.observable_mask & ~previous_team_map
        )
        total = int(np.count_nonzero(newly_explored))
        return total, per_agent_new

    @staticmethod
    def _fov_overlap_cells(fov_sets: Sequence[Set[Coordinate]]) -> int:
        if not fov_sets:
            return 0
        total = sum(len(visible) for visible in fov_sets)
        union_size = len(set().union(*fov_sets))
        return int(max(0, total - union_size))

    # ------------------------------------------------------------------
    # Synchronous movement and reward
    # ------------------------------------------------------------------
    def _is_free(self, position: Coordinate) -> bool:
        x, y = position
        return (
            0 <= x < self.size
            and 0 <= y < self.size
            and not self.wall_map[y, x]
        )

    def _resolve_movement(self, actions: Sequence[int]) -> MovementResult:
        raw_actions = np.asarray(actions)
        if raw_actions.shape != (self.n_agents,):
            raise ValueError(
                f"actions must have shape ({self.n_agents},), got {raw_actions.shape}"
            )
        if not np.issubdtype(raw_actions.dtype, np.integer):
            raise TypeError("actions must use an integer dtype")
        actions_array = raw_actions.astype(np.int64, copy=False)
        if np.any((actions_array < 0) | (actions_array >= N_ACTIONS)):
            raise ValueError(f"actions must be in [0, {N_ACTIONS - 1}]")

        old_positions = tuple(self.agent_positions)
        directions = list(self.agent_dirs)
        proposals = list(old_positions)
        collisions = np.zeros(self.n_agents, dtype=bool)
        turned = np.zeros(self.n_agents, dtype=bool)
        forward_attempts = actions_array == ACTION_FORWARD

        # Build all proposals from the same state before resolving conflicts.
        for index, action in enumerate(actions_array.tolist()):
            if action == ACTION_LEFT:
                directions[index] = (directions[index] - 1) % 4
                turned[index] = True
            elif action == ACTION_RIGHT:
                directions[index] = (directions[index] + 1) % 4
                turned[index] = True
            else:
                dx, dy = _DIRECTION_VECTORS[self.agent_dirs[index]]
                candidate = (
                    old_positions[index][0] + dx,
                    old_positions[index][1] + dy,
                )
                if self._is_free(candidate):
                    proposals[index] = candidate
                else:
                    collisions[index] = True

        # All agents targeting the same cell remain in place.
        by_target: Dict[Coordinate, List[int]] = {}
        for index, target in enumerate(proposals):
            if target != old_positions[index]:
                by_target.setdefault(target, []).append(index)
        for indices in by_target.values():
            if len(indices) > 1:
                for index in indices:
                    proposals[index] = old_positions[index]
                    collisions[index] = True

        # Direct position swaps are not allowed.
        for first in range(self.n_agents):
            for second in range(first + 1, self.n_agents):
                if (
                    proposals[first] == old_positions[second]
                    and proposals[second] == old_positions[first]
                    and proposals[first] != old_positions[first]
                ):
                    proposals[first] = old_positions[first]
                    proposals[second] = old_positions[second]
                    collisions[first] = True
                    collisions[second] = True

        # Resolve movement into a cell whose previous occupant does not leave.
        changed = True
        while changed:
            changed = False
            for index, target in enumerate(proposals):
                if target == old_positions[index]:
                    continue
                for occupant, occupied_cell in enumerate(old_positions):
                    if (
                        index != occupant
                        and target == occupied_cell
                        and proposals[occupant] == occupied_cell
                    ):
                        proposals[index] = old_positions[index]
                        collisions[index] = True
                        changed = True
                        break

        moved = np.asarray(
            [proposals[index] != old_positions[index] for index in range(self.n_agents)],
            dtype=bool,
        )
        return MovementResult(
            positions=tuple(proposals),
            directions=tuple(directions),
            collisions=tuple(bool(value) for value in collisions),
            moved=tuple(bool(value) for value in moved),
            turned=tuple(bool(value) for value in turned),
            forward_attempts=tuple(bool(value) for value in forward_attempts),
        )

    def step(
        self, actions: Sequence[int]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool, bool, Dict[str, object]]:
        if self._episode_done:
            raise RuntimeError("episode is complete; call reset() before step()")
        movement = self._resolve_movement(actions)
        self.agent_positions = list(movement.positions)
        self.agent_dirs = list(movement.directions)
        self.step_count += 1

        collisions = np.asarray(movement.collisions, dtype=bool)
        moved = np.asarray(movement.moved, dtype=bool)
        turned = np.asarray(movement.turned, dtype=bool)
        forward_attempts = np.asarray(movement.forward_attempts, dtype=bool)
        self.collisions_total += int(np.count_nonzero(collisions))
        self.moves_total += int(np.count_nonzero(moved))
        self.turns_total += int(np.count_nonzero(turned))
        self.forward_attempts_total += int(np.count_nonzero(forward_attempts))
        for index, (x, y) in enumerate(self.agent_positions):
            self.visited_maps[index, y, x] = True

        # One synchronous transition: move all agents, compute all FOVs, update
        # all beliefs once, then generate every observation and per-agent rewards.
        fov_sets = [self._compute_fov(index) for index in range(self.n_agents)]
        new_cells_total, new_cells_per_agent = self._apply_fov_updates(fov_sets)
        overlap_cells = self._fov_overlap_cells(fov_sets)
        terminated = self.coverage >= self.target_coverage
        truncated = self.step_count >= self.max_steps and not terminated
        self._episode_done = bool(terminated or truncated)
        per_agent_rewards = self._compute_team_reward(
            new_cells_per_agent=new_cells_per_agent,
            collisions=collisions,
            fov_sets=fov_sets,
            terminated=terminated,
        )
        team_reward = float(np.mean(per_agent_rewards))

        self._last_metrics = self._build_metrics(
            new_cells=new_cells_total,
            collisions=collisions,
            moved=moved,
            turned=turned,
            forward_attempts=forward_attempts,
            overlap_cells=overlap_cells,
            team_reward=team_reward,
            terminated=terminated,
            truncated=truncated,
        )
        return self.observation(), per_agent_rewards, bool(terminated), bool(
            truncated
        ), {
            "team_reward": team_reward,
            "new_cells": new_cells_total,
            "metrics": dict(self._last_metrics),
        }

    def _compute_team_reward(
        self,
        new_cells_per_agent: np.ndarray,
        collisions: np.ndarray,
        fov_sets: Sequence[Set[Coordinate]],
        terminated: bool,
    ) -> np.ndarray:
        config = self.reward_config
        target_count = max(1, self.target_cell_count)
        info_gain_per_agent = new_cells_per_agent.astype(np.float32) / target_count
        collision_penalties = collisions.astype(np.float32) * config.collision_penalty
        redundancy = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            others = [s for j, s in enumerate(fov_sets) if j != i]
            overlap_i = len(fov_sets[i] & set().union(*others)) if others else 0
            visible_i = max(1, len(fov_sets[i]))
            redundancy[i] = config.redundancy_penalty * (overlap_i / visible_i)
        rewards = (
            config.information_gain_scale * info_gain_per_agent
            - config.step_penalty
            - collision_penalties
            - redundancy
        )
        for threshold, bonus in (
            (0.25, config.milestone_25_bonus),
            (0.50, config.milestone_50_bonus),
            (0.75, config.milestone_75_bonus),
        ):
            if self.coverage >= threshold and threshold not in self._milestones_reached:
                self._milestones_reached.add(threshold)
                rewards += bonus / self.n_agents
        if terminated:
            rewards += config.completion_bonus / self.n_agents
        return rewards.astype(np.float32)

    # ------------------------------------------------------------------
    # Observation, metrics, and rendering
    # ------------------------------------------------------------------
    @property
    def target_cell_count(self) -> int:
        return int(np.count_nonzero(self.observable_mask))

    @property
    def explored_target_count(self) -> int:
        return int(
            np.count_nonzero(self.team_explored_map & self.observable_mask)
        )

    @property
    def coverage(self) -> float:
        denominator = self.target_cell_count
        return (
            float(self.explored_target_count / denominator)
            if denominator
            else 0.0
        )

    def observation(self) -> Dict[str, np.ndarray]:
        images = np.empty(
            (self.n_agents, self.size, self.size, OBS_CHANNELS),
            dtype=np.float32,
        )
        for index in range(self.n_agents):
            known = self.known_maps[index]
            images[index, ..., 0] = known == UNKNOWN
            images[index, ..., 1] = known == FREE
            images[index, ..., 2] = known == WALL
            images[index, ..., 3] = self.current_fov_maps[index]
            images[index, ..., 4] = self.visited_maps[index]

        aux = np.zeros((self.n_agents, AUX_DIM), dtype=np.float32)
        coordinate_scale = float(max(1, self.size - 1))
        for index, ((x, y), direction) in enumerate(
            zip(self.agent_positions, self.agent_dirs)
        ):
            aux[index, 0] = x / coordinate_scale
            aux[index, 1] = y / coordinate_scale
            aux[index, 2 + direction] = 1.0
        return {"image": images, "aux": aux}

    def _build_metrics(
        self,
        new_cells: int,
        collisions: np.ndarray,
        moved: np.ndarray,
        turned: np.ndarray,
        forward_attempts: np.ndarray,
        overlap_cells: int,
        team_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, object]:
        action_denominator = max(1, self.step_count * self.n_agents)
        local_known_fraction = [
            float(
                np.count_nonzero(self.known_maps[index] != UNKNOWN)
                / max(1, self.target_cell_count)
            )
            for index in range(self.n_agents)
        ]
        return {
            "step": int(self.step_count),
            "coverage": float(self.coverage),
            "explored_cells": int(self.explored_target_count),
            "target_cells": int(self.target_cell_count),
            "new_cells": int(new_cells),
            "team_reward": float(team_reward),
            "success": bool(terminated),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "collisions_step": int(np.count_nonzero(collisions)),
            "collisions_total": int(self.collisions_total),
            "collision_rate": float(self.collisions_total / action_denominator),
            "moves_step": int(np.count_nonzero(moved)),
            "moves_total": int(self.moves_total),
            "turns_step": int(np.count_nonzero(turned)),
            "turns_total": int(self.turns_total),
            "forward_attempts_step": int(np.count_nonzero(forward_attempts)),
            "forward_attempts_total": int(self.forward_attempts_total),
            "fov_overlap_cells": int(overlap_cells),
            "local_known_fraction": local_known_fraction,
        }

    def get_metrics(self) -> Dict[str, object]:
        return dict(self._last_metrics)

    def render(self, cell_size: int = 24) -> np.ndarray:
        cell_size = max(4, int(cell_size))
        image = np.zeros(
            (self.size * cell_size, self.size * cell_size, 3), dtype=np.uint8
        )
        image[:] = (24, 24, 24)
        for y in range(self.size):
            for x in range(self.size):
                y0, y1 = y * cell_size, (y + 1) * cell_size
                x0, x1 = x * cell_size, (x + 1) * cell_size
                if self.team_explored_map[y, x] and self.wall_map[y, x]:
                    color = (95, 95, 95)
                elif self.team_explored_map[y, x]:
                    color = (215, 215, 215)
                elif self.wall_map[y, x]:
                    color = (38, 38, 38)
                else:
                    color = (28, 28, 28)
                image[y0:y1, x0:x1] = color
                image[y0:y0 + 1, x0:x1] = (55, 55, 55)
                image[y0:y1, x0:x0 + 1] = (55, 55, 55)

        colors = (
            (225, 65, 65),
            (65, 135, 225),
            (65, 190, 95),
            (230, 180, 45),
        )
        for index, ((x, y), direction) in enumerate(
            zip(self.agent_positions, self.agent_dirs)
        ):
            y0, y1 = y * cell_size, (y + 1) * cell_size
            x0, x1 = x * cell_size, (x + 1) * cell_size
            margin = max(1, cell_size // 6)
            image[
                y0 + margin:y1 - margin,
                x0 + margin:x1 - margin,
            ] = colors[index % len(colors)]
            center_x = (x0 + x1) // 2
            center_y = (y0 + y1) // 2
            dx, dy = _DIRECTION_VECTORS[direction]
            tip_x = int(center_x + dx * cell_size * 0.32)
            tip_y = int(center_y + dy * cell_size * 0.32)
            radius = max(1, cell_size // 10)
            image[
                max(y0, tip_y - radius):min(y1, tip_y + radius + 1),
                max(x0, tip_x - radius):min(x1, tip_x + radius + 1),
            ] = (255, 255, 255)
        return image


__all__ = [
    "ACTION_FORWARD",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "AUX_DIM",
    "FREE",
    "N_ACTIONS",
    "OBS_CHANNELS",
    "OBSERVATION_CHANNELS",
    "RewardConfig",
    "UNKNOWN",
    "WALL",
    "CooperativeSLAMCore",
]
