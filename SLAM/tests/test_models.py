from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from SLAM.maa2c import MAA2C
from SLAM.slam_baselines import (
    count_trainable_params,
    generate_actor,
    generate_critic_fctde,
    generate_critic_sctde,
)


def test_actor_and_classical_critics_accept_image_and_aux() -> None:
    obs_shape = (8, 8, 5)
    actor = generate_actor(obs_shape=obs_shape)
    sctde = generate_critic_sctde(n_agents=2, obs_shape=obs_shape)
    fctde = generate_critic_fctde(n_agents=2, obs_shape=obs_shape)

    local_images = tf.zeros((2, *obs_shape), dtype=tf.float32)
    local_aux = tf.zeros((2, 6), dtype=tf.float32)
    joint_images = tf.zeros((3, 2, *obs_shape), dtype=tf.float32)
    joint_aux = tf.zeros((3, 2, 6), dtype=tf.float32)

    assert actor([local_images, local_aux]).shape == (2, 3)
    assert sctde([joint_images, joint_aux]).shape == (3, 2)
    assert fctde([joint_images, joint_aux]).shape == (3, 2)
    np.testing.assert_allclose(
        tf.reduce_sum(actor([local_images, local_aux]), axis=-1).numpy(),
        np.ones(2),
        rtol=1e-5,
    )


def test_sctde_and_fctde_are_structurally_distinct() -> None:
    obs_shape = (8, 8, 5)
    sctde = generate_critic_sctde(n_agents=2, obs_shape=obs_shape)
    fctde = generate_critic_fctde(n_agents=2, obs_shape=obs_shape)
    sctde_names = {layer.name for layer in sctde.layers}
    fctde_names = {layer.name for layer in fctde.layers}
    assert any(name.startswith("sctde-agent-0") for name in sctde_names)
    assert any(name.startswith("sctde-agent-1") for name in sctde_names)
    assert "fctde-joint-state" in fctde_names
    assert not any(name.startswith("sctde-agent") for name in fctde_names)
    assert count_trainable_params(sctde) != count_trainable_params(fctde)


def test_both_classical_critics_connect_joint_aux_to_value() -> None:
    obs_shape = (8, 8, 5)
    for builder in (generate_critic_sctde, generate_critic_fctde):
        model = builder(n_agents=2, obs_shape=obs_shape)
        assert len(model.inputs) == 2
        assert model.inputs[1].shape.as_list() == [None, 2, 6]
        # The auxiliary input must have an outbound path in the functional graph.
        assert model.get_layer("joint-aux") is not None


def test_gae_stops_at_true_terminal_and_bootstraps_truncation() -> None:
    algorithm = object.__new__(MAA2C)
    algorithm.gamma = 0.9
    algorithm.gae_lambda = 1.0
    rewards = np.asarray([[1.0, 0.5], [1.0, 0.5]], dtype=np.float32)
    values = np.asarray([[0.5, 0.25], [0.5, 0.25]], dtype=np.float32)
    next_values = np.asarray([[0.5, 0.25], [10.0, 5.0]], dtype=np.float32)

    terminal_advantages, terminal_returns = algorithm._gae_targets(
        rewards,
        np.asarray([0.0, 1.0], dtype=np.float32),
        values,
        next_values,
    )
    truncated_advantages, truncated_returns = algorithm._gae_targets(
        rewards,
        np.asarray([0.0, 0.0], dtype=np.float32),
        values,
        next_values,
    )
    assert terminal_returns[-1, 0] == pytest.approx(1.0)
    assert terminal_returns[-1, 1] == pytest.approx(0.5)
    assert truncated_returns[-1, 0] == pytest.approx(10.0)
    assert truncated_returns[-1, 1] == pytest.approx(5.0)
    assert truncated_advantages[-1, 0] > terminal_advantages[-1, 0]

