"""Gymnasium interfaces for the corrected cooperative SLAM environment."""
from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry

try:
    from .slam_core import (
        ACTION_FORWARD,
        ACTION_LEFT,
        ACTION_RIGHT,
        AUX_DIM,
        N_ACTIONS,
        OBS_CHANNELS,
        OBSERVATION_CHANNELS,
        CooperativeSLAMCore,
        RewardConfig,
    )
except ImportError:  # Support running directly from the SLAM directory.
    from slam_core import (
        ACTION_FORWARD,
        ACTION_LEFT,
        ACTION_RIGHT,
        AUX_DIM,
        N_ACTIONS,
        OBS_CHANNELS,
        OBSERVATION_CHANNELS,
        CooperativeSLAMCore,
        RewardConfig,
    )

_ENV_PATTERN = re.compile(
    r"^MiniGrid-(Empty|Hard|Random)SLAM-(\d+)x(\d+)-v0$",
    flags=re.IGNORECASE,
)


def parse_env_id(env_id: str) -> Tuple[str, int]:
    """Parse a registered environment ID into ``(maze_type, size)``."""
    match = _ENV_PATTERN.fullmatch(str(env_id).strip())
    if match is None:
        raise ValueError(
            "invalid environment ID; expected e.g. "
            "MiniGrid-RandomSLAM-8x8-v0"
        )
    width = int(match.group(2))
    height = int(match.group(3))
    if width != height:
        raise ValueError("only square maps are supported")
    return match.group(1).lower(), width


class MultiAgentSLAMEnv(gym.Env):
    """Joint-action Gymnasium environment for cooperative SLAM.

    The action is a ``MultiDiscrete`` vector with one action per agent.  The
    reward is one scalar team reward, making the environment a valid standard
    Gymnasium joint-control environment.

    Observation ``image`` has shape ``(agents, H, W, 5)`` with binary channels
    ``unknown``, ``known_free``, ``known_wall``, ``current_fov``, and
    ``visited_by_self``.  Observation ``aux`` has shape ``(agents, 6)`` with
    normalized x/y and a four-way heading one-hot vector.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        size: int = 8,
        max_steps: int = 100,
        n_agents: int = 2,
        maze_type: str = "random",
        obstacle_density: float = 0.15,
        fov_range: int = 4,
        fov_angle: float = np.pi / 2.0,
        target_coverage: float = 0.95,
        reward_config: Optional[RewardConfig] = None,
        agent_start_positions: Optional[Sequence[Tuple[int, int]]] = None,
        agent_start_dirs: Optional[Sequence[int]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        self.core = CooperativeSLAMCore(
            size=size,
            n_agents=n_agents,
            max_steps=max_steps,
            maze_type=maze_type,
            obstacle_density=obstacle_density,
            fov_range=fov_range,
            fov_angle=fov_angle,
            target_coverage=target_coverage,
            reward_config=reward_config,
            agent_start_positions=agent_start_positions,
            agent_start_dirs=agent_start_dirs,
        )
        self.action_space = spaces.MultiDiscrete(
            np.full((self.n_agents,), N_ACTIONS, dtype=np.int64)
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.n_agents, self.size, self.size, OBS_CHANNELS),
                    dtype=np.float32,
                ),
                "aux": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.n_agents, AUX_DIM),
                    dtype=np.float32,
                ),
            }
        )

    @property
    def n_agents(self) -> int:
        return self.core.n_agents

    @property
    def size(self) -> int:
        return self.core.size

    @property
    def max_steps(self) -> int:
        return self.core.max_steps

    @property
    def maze_type(self) -> str:
        return self.core.maze_type

    @property
    def agent_positions(self):
        return self.core.agent_positions

    @property
    def agent_dirs(self):
        return self.core.agent_dirs

    @property
    def wall_map(self):
        return self.core.wall_map

    @property
    def known_maps(self):
        return self.core.known_maps

    @property
    def explored_map(self):
        """Compatibility alias for the team union map."""
        return self.core.team_explored_map

    @property
    def current_fovs(self):
        coordinate_sets = []
        for index in range(self.n_agents):
            ys, xs = np.where(self.core.current_fov_maps[index])
            coordinate_sets.append(
                [(int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())]
            )
        return coordinate_sets

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        del options
        return self.core.reset(seed=seed)

    def step(self, action):
        return self.core.step(action)

    def get_metrics(self) -> Dict[str, object]:
        return self.core.get_metrics()

    def render(self):
        return self.core.render()

    def close(self) -> None:
        return None


class SingleAgentSLAMEnv(gym.Env):
    """Single-agent view used by the registered environment IDs."""

    metadata = MultiAgentSLAMEnv.metadata

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        kwargs["n_agents"] = 1
        self.joint_env = MultiAgentSLAMEnv(*args, **kwargs)
        self.render_mode = self.joint_env.render_mode
        self.action_space = spaces.Discrete(N_ACTIONS)
        size = self.joint_env.size
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(size, size, OBS_CHANNELS),
                    dtype=np.float32,
                ),
                "aux": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(AUX_DIM,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation, info = self.joint_env.reset(seed=seed, options=options)
        return {
            "image": observation["image"][0],
            "aux": observation["aux"][0],
        }, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.joint_env.step(
            [int(action)]
        )
        return (
            {
                "image": observation["image"][0],
                "aux": observation["aux"][0],
            },
            float(reward[0]),
            terminated,
            truncated,
            info,
        )

    def render(self):
        return self.joint_env.render()

    def close(self):
        return self.joint_env.close()

    def get_metrics(self):
        return self.joint_env.get_metrics()


SLAMEnv = SingleAgentSLAMEnv


class SLAMEnv6x6(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "empty"
        super().__init__(size=6, **kwargs)


class SLAMEnv8x8(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "empty"
        super().__init__(size=8, **kwargs)


class SLAMEnv16x16(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "empty"
        super().__init__(size=16, **kwargs)


class SLAMEnv32x32(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "empty"
        super().__init__(size=32, **kwargs)


class HardSLAMEnv6x6(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "hard"
        super().__init__(size=6, **kwargs)


class HardSLAMEnv8x8(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "hard"
        super().__init__(size=8, **kwargs)


class HardSLAMEnv16x16(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "hard"
        super().__init__(size=16, **kwargs)


class HardSLAMEnv32x32(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "hard"
        super().__init__(size=32, **kwargs)


class RandomSLAMEnv6x6(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "random"
        super().__init__(size=6, **kwargs)


class RandomSLAMEnv8x8(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "random"
        super().__init__(size=8, **kwargs)


class RandomSLAMEnv16x16(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "random"
        super().__init__(size=16, **kwargs)


class RandomSLAMEnv32x32(SingleAgentSLAMEnv):
    def __init__(self, **kwargs):
        kwargs["maze_type"] = "random"
        super().__init__(size=32, **kwargs)


_REGISTRATIONS = {
    "MiniGrid-EmptySLAM-6x6-v0": "SLAMEnv6x6",
    "MiniGrid-EmptySLAM-8x8-v0": "SLAMEnv8x8",
    "MiniGrid-EmptySLAM-16x16-v0": "SLAMEnv16x16",
    "MiniGrid-EmptySLAM-32x32-v0": "SLAMEnv32x32",
    "MiniGrid-HardSLAM-6x6-v0": "HardSLAMEnv6x6",
    "MiniGrid-HardSLAM-8x8-v0": "HardSLAMEnv8x8",
    "MiniGrid-HardSLAM-16x16-v0": "HardSLAMEnv16x16",
    "MiniGrid-HardSLAM-32x32-v0": "HardSLAMEnv32x32",
    "MiniGrid-RandomSLAM-6x6-v0": "RandomSLAMEnv6x6",
    "MiniGrid-RandomSLAM-8x8-v0": "RandomSLAMEnv8x8",
    "MiniGrid-RandomSLAM-16x16-v0": "RandomSLAMEnv16x16",
    "MiniGrid-RandomSLAM-32x32-v0": "RandomSLAMEnv32x32",
}
for _environment_id, _class_name in _REGISTRATIONS.items():
    if _environment_id not in registry:
        register(
            id=_environment_id,
            entry_point=f"{__name__}:{_class_name}",
        )


def make_multi_agent_env_from_id(
    env_id: str,
    n_agents: int = 2,
    max_steps: int = 100,
    **kwargs,
) -> MultiAgentSLAMEnv:
    maze_type, size = parse_env_id(env_id)
    return MultiAgentSLAMEnv(
        size=size,
        max_steps=max_steps,
        n_agents=n_agents,
        maze_type=maze_type,
        **kwargs,
    )


class SharedRoomVecEnv:
    """Legacy adapter for the former vector-like API.

    This is not a Gymnasium ``VectorEnv``: the agents are coupled in one joint
    environment.  New code should use :class:`MultiAgentSLAMEnv` directly.
    """

    def __init__(self, env: MultiAgentSLAMEnv):
        self.env = env
        self.num_envs = env.n_agents
        self.single_action_space = spaces.Discrete(N_ACTIONS)
        self.action_space = env.action_space
        self.observation_space = spaces.Tuple(
            (env.observation_space["image"], env.observation_space["aux"])
        )

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        return (observation["image"], observation["aux"]), info

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        rewards = np.asarray(reward, dtype=np.float32)
        terminations = np.full(self.num_envs, terminated, dtype=bool)
        truncations = np.full(self.num_envs, truncated, dtype=bool)
        infos = [dict(info) for _ in range(self.num_envs)]
        return (
            (observation["image"], observation["aux"]),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


__all__ = [
    "ACTION_FORWARD",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "AUX_DIM",
    "N_ACTIONS",
    "OBS_CHANNELS",
    "OBSERVATION_CHANNELS",
    "RewardConfig",
    "MultiAgentSLAMEnv",
    "SingleAgentSLAMEnv",
    "SLAMEnv",
    "SharedRoomVecEnv",
    "make_multi_agent_env_from_id",
    "parse_env_id",
]
