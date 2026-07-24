from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env

from SLAM import slam


def test_multi_agent_environment_passes_gymnasium_checker() -> None:
    env = slam.MultiAgentSLAMEnv(size=8, n_agents=2, max_steps=10)
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=100)
    assert env.observation_space.contains(observation)
    action = np.asarray([0, 1], dtype=np.int64)
    next_observation, reward, terminated, truncated, step_info = env.step(action)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, np.ndarray)
    assert reward.shape == (2,)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "metrics" in info and "metrics" in step_info
    env.close()


def test_registered_single_agent_environment_matches_declared_spaces() -> None:
    env = gym.make("MiniGrid-RandomSLAM-8x8-v0", max_steps=10)
    observation, _ = env.reset(seed=101)
    assert env.observation_space.contains(observation)
    next_observation, reward, terminated, truncated, _ = env.step(1)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_env_id_controls_maze_type_in_multi_agent_factory() -> None:
    empty = slam.make_multi_agent_env_from_id(
        "MiniGrid-EmptySLAM-8x8-v0", max_steps=10
    )
    hard = slam.make_multi_agent_env_from_id(
        "MiniGrid-HardSLAM-8x8-v0", max_steps=10
    )
    random = slam.make_multi_agent_env_from_id(
        "MiniGrid-RandomSLAM-8x8-v0", max_steps=10
    )
    assert empty.maze_type == "empty"
    assert hard.maze_type == "hard"
    assert random.maze_type == "random"
    empty.close()
    hard.close()
    random.close()


def test_legacy_adapter_declares_tuple_observation_and_team_reward_vector() -> None:
    env = slam.MultiAgentSLAMEnv(size=8, n_agents=2, max_steps=10)
    adapter = slam.SharedRoomVecEnv(env)
    observation, _ = adapter.reset(seed=102)
    assert adapter.observation_space.contains(observation)
    _, rewards, terminations, truncations, infos = adapter.step([1, 2])
    assert rewards.shape == (2,)
    assert terminations.shape == (2,)
    assert truncations.shape == (2,)
    assert len(infos) == 2
    adapter.close()
