from __future__ import annotations

import numpy as np
import pytest

from SLAM.slam_core import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    FREE,
    OBS_CHANNELS,
    UNKNOWN,
    WALL,
    CooperativeSLAMCore,
)


def assert_semantic_observation(observation, n_agents: int, size: int) -> None:
    images = observation["image"]
    aux = observation["aux"]
    assert images.shape == (n_agents, size, size, OBS_CHANNELS)
    assert aux.shape == (n_agents, 6)
    assert images.dtype == np.float32
    assert aux.dtype == np.float32
    assert np.all((images == 0.0) | (images == 1.0))
    # unknown/free/wall are a categorical partition; FOV/visited are overlays.
    np.testing.assert_allclose(images[..., :3].sum(axis=-1), 1.0)
    np.testing.assert_allclose(aux[:, 2:].sum(axis=-1), 1.0)


@pytest.mark.parametrize("maze_type", ["empty", "hard", "random"])
@pytest.mark.parametrize("size", [6, 8, 16])
def test_reset_is_valid_and_free_space_is_connected(maze_type: str, size: int) -> None:
    core = CooperativeSLAMCore(size=size, maze_type=maze_type, max_steps=50)
    observation, info = core.reset(seed=1234)
    assert_semantic_observation(observation, n_agents=2, size=size)
    assert info["seed"] == 1234
    assert core._free_cells_connected(core.wall_map)
    assert not any(core.wall_map[y, x] for x, y in core.agent_positions)
    reachable = core._reachable_free_mask(core.agent_positions[0])
    assert np.all(reachable[~core.wall_map])
    assert core.target_cell_count > 0
    assert 0.0 <= core.coverage <= 1.0


def test_same_seed_reproduces_map_state_and_observation() -> None:
    first = CooperativeSLAMCore(size=16, maze_type="random")
    second = CooperativeSLAMCore(size=16, maze_type="random")
    first_observation, _ = first.reset(seed=912_441)
    second_observation, _ = second.reset(seed=912_441)
    np.testing.assert_array_equal(first.wall_map, second.wall_map)
    np.testing.assert_array_equal(first.observable_mask, second.observable_mask)
    np.testing.assert_array_equal(first_observation["image"], second_observation["image"])
    np.testing.assert_array_equal(first_observation["aux"], second_observation["aux"])
    assert first.agent_positions == second.agent_positions
    assert first.agent_dirs == second.agent_dirs


def test_ground_truth_map_is_not_leaked_into_local_observation() -> None:
    core = CooperativeSLAMCore(size=16, maze_type="random", fov_range=3)
    observation, _ = core.reset(seed=55)
    local_known = core.known_maps[0]
    hidden_wall_locations = np.argwhere(core.wall_map & (local_known == UNKNOWN))
    assert hidden_wall_locations.size > 0
    hidden_y, hidden_x = map(int, hidden_wall_locations[0])
    assert observation["image"][0, hidden_y, hidden_x, 0] == 1.0
    assert observation["image"][0, hidden_y, hidden_x, 2] == 0.0


def test_coordinate_convention_is_y_x_without_transpose_mismatch() -> None:
    core = CooperativeSLAMCore(
        size=6,
        n_agents=1,
        maze_type="empty",
        fov_range=5,
        agent_start_positions=[(2, 2)],
        agent_start_dirs=[3],  # north
    )
    observation, _ = core.reset(seed=1)
    # The north boundary wall at Cartesian (x=2, y=0) is stored at [0, 2].
    assert core.wall_map[0, 2]
    assert core.known_maps[0, 0, 2] == WALL
    assert observation["image"][0, 0, 2, 2] == 1.0
    assert observation["image"][0, 2, 0, 2] == 0.0


def test_agents_keep_distinct_local_maps() -> None:
    core = CooperativeSLAMCore(size=16, maze_type="empty", fov_range=3)
    observation, _ = core.reset(seed=2)
    assert not np.array_equal(core.known_maps[0], core.known_maps[1])
    np.testing.assert_array_equal(
        observation["image"][0, ..., 0], core.known_maps[0] == UNKNOWN
    )
    np.testing.assert_array_equal(
        observation["image"][1, ..., 0], core.known_maps[1] == UNKNOWN
    )


def test_turning_is_not_a_collision_or_standing_still_failure() -> None:
    core = CooperativeSLAMCore(
        size=8,
        maze_type="empty",
        agent_start_positions=[(2, 2), (5, 5)],
        agent_start_dirs=[0, 2],
    )
    core.reset(seed=3)
    before = tuple(core.agent_positions)
    _, reward, terminated, truncated, info = core.step([ACTION_LEFT, ACTION_RIGHT])
    assert tuple(core.agent_positions) == before
    metrics = info["metrics"]
    assert metrics["turns_step"] == 2
    assert metrics["collisions_step"] == 0
    assert metrics["forward_attempts_step"] == 0
    assert np.all(reward > -0.2)
    assert not terminated
    assert not truncated


def test_same_target_conflict_is_synchronous_and_symmetric() -> None:
    core = CooperativeSLAMCore(
        size=7,
        maze_type="empty",
        agent_start_positions=[(2, 3), (4, 3)],
        agent_start_dirs=[0, 2],
    )
    core.reset(seed=4)
    _, _, _, _, info = core.step([ACTION_FORWARD, ACTION_FORWARD])
    assert core.agent_positions == [(2, 3), (4, 3)]
    assert info["metrics"]["collisions_step"] == 2
    assert info["metrics"]["moves_step"] == 0


def test_direct_swap_is_blocked_for_both_agents() -> None:
    core = CooperativeSLAMCore(
        size=7,
        maze_type="empty",
        agent_start_positions=[(2, 3), (3, 3)],
        agent_start_dirs=[0, 2],
    )
    core.reset(seed=5)
    _, _, _, _, info = core.step([ACTION_FORWARD, ACTION_FORWARD])
    assert core.agent_positions == [(2, 3), (3, 3)]
    assert info["metrics"]["collisions_step"] == 2


def test_all_agent_observations_are_generated_after_one_joint_transition() -> None:
    core = CooperativeSLAMCore(
        size=8,
        maze_type="empty",
        agent_start_positions=[(2, 2), (5, 5)],
        agent_start_dirs=[0, 2],
    )
    core.reset(seed=6)
    observation, _, _, _, _ = core.step([ACTION_FORWARD, ACTION_LEFT])
    # Both returned current-FOV overlays match the final core state; neither is
    # a snapshot taken before the other agent's update.
    np.testing.assert_array_equal(
        observation["image"][..., 3], core.current_fov_maps.astype(np.float32)
    )
    np.testing.assert_array_equal(
        core.team_explored_map, np.any(core.known_maps != UNKNOWN, axis=0)
    )


def test_reward_is_normalized_by_observable_map_size() -> None:
    rewards = []
    for size in (8, 16, 32):
        core = CooperativeSLAMCore(size=size, maze_type="empty", max_steps=50)
        core.reset(seed=7)
        _, reward, _, _, _ = core.step([ACTION_LEFT, ACTION_RIGHT])
        rewards.append(reward)
    for r in rewards:
        assert isinstance(r, np.ndarray)
        assert r.shape == (2,)
        assert np.all((-1.0 < r) & (r < 2.0))


def test_episode_truncates_at_configured_horizon() -> None:
    core = CooperativeSLAMCore(
        size=16,
        maze_type="empty",
        max_steps=2,
        target_coverage=1.0,
    )
    core.reset(seed=8)
    _, _, terminated, truncated, _ = core.step([ACTION_LEFT, ACTION_RIGHT])
    assert not terminated
    assert not truncated
    _, _, terminated, truncated, _ = core.step([ACTION_RIGHT, ACTION_LEFT])
    assert not terminated
    assert truncated


def test_known_map_uses_semantic_categories_not_scaled_object_ids() -> None:
    core = CooperativeSLAMCore(size=8, maze_type="empty")
    observation, _ = core.reset(seed=9)
    assert set(np.unique(core.known_maps).tolist()).issubset(
        {int(UNKNOWN), int(FREE), int(WALL)}
    )
    assert set(np.unique(observation["image"]).tolist()).issubset({0.0, 1.0})


def test_step_after_episode_end_requires_reset() -> None:
    core = CooperativeSLAMCore(
        size=16,
        maze_type="empty",
        max_steps=1,
        target_coverage=1.0,
    )
    core.reset(seed=10)
    _, _, terminated, truncated, _ = core.step([ACTION_LEFT, ACTION_RIGHT])
    assert terminated or truncated
    with pytest.raises(RuntimeError, match="call reset"):
        core.step([ACTION_LEFT, ACTION_RIGHT])


def test_negative_reward_coefficients_are_rejected() -> None:
    from SLAM.slam_core import RewardConfig

    with pytest.raises(ValueError, match="must be non-negative"):
        CooperativeSLAMCore(
            size=8,
            reward_config=RewardConfig(collision_penalty=-1.0),
        )


def test_every_visible_cell_belongs_to_the_reachable_observable_target() -> None:
    core = CooperativeSLAMCore(size=16, maze_type="random", fov_range=4)
    core.reset(seed=11)
    for _ in range(12):
        core.step([ACTION_LEFT, ACTION_RIGHT])
        assert np.all(
            np.any(core.current_fov_maps, axis=0) <= core.observable_mask
        )


def test_non_integer_actions_are_rejected_without_silent_casting() -> None:
    core = CooperativeSLAMCore(size=8, maze_type="empty")
    core.reset(seed=12)
    with pytest.raises(TypeError, match="integer dtype"):
        core.step(np.asarray([0.0, 1.0], dtype=np.float32))


def test_non_finite_reward_coefficients_are_rejected() -> None:
    from SLAM.slam_core import RewardConfig

    with pytest.raises(ValueError, match="finite"):
        CooperativeSLAMCore(
            size=8,
            reward_config=RewardConfig(information_gain_scale=float("nan")),
        )


def test_duplicate_or_out_of_bounds_start_positions_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        CooperativeSLAMCore(
            size=8,
            n_agents=2,
            agent_start_positions=[(2, 2), (2, 2)],
        )
    with pytest.raises(ValueError, match="interior"):
        CooperativeSLAMCore(
            size=8,
            n_agents=1,
            agent_start_positions=[(0, 2)],
        )
