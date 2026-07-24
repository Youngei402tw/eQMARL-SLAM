from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("cirq")
pytest.importorskip("tensorflow_quantum")

import tensorflow as tf

from SLAM.quantum_layers import CentralVariationalPQC, PartiteVariationalPQC
from SLAM.slam_baselines import (
    generate_critic_eqmarl_psi_plus,
    generate_critic_qfctde,
)


def test_quantum_layers_forward_and_matched_readout_shapes() -> None:
    split = PartiteVariationalPQC(
        n_agents=2,
        d_qubits=2,
        n_layers=1,
        input_entanglement=True,
        entanglement_type="psi+",
    )
    central = CentralVariationalPQC(n_agents=2, d_qubits=2, n_layers=1)
    split_output = split(tf.zeros((2, 2, 2, 3), dtype=tf.float32))
    central_output = central(tf.zeros((2, 4, 3), dtype=tf.float32))
    assert split_output.shape == central_output.shape
    # two per-agent partition Z + two partition X + two partition Y observables
    assert split_output.shape == (2, 6)


def test_quantum_critics_accept_the_same_joint_state_interface() -> None:
    obs_shape = (6, 6, 5)
    joint_images = tf.zeros((2, 2, *obs_shape), dtype=tf.float32)
    joint_aux = tf.zeros((2, 2, 6), dtype=tf.float32)
    eqmarl = generate_critic_eqmarl_psi_plus(
        n_agents=2,
        d_qubits=2,
        n_layers=1,
        obs_shape=obs_shape,
    )
    qfctde = generate_critic_qfctde(
        n_agents=2,
        d_qubits=2,
        n_layers=1,
        obs_shape=obs_shape,
    )
    assert eqmarl([joint_images, joint_aux]).shape == (2, 2)
    assert qfctde([joint_images, joint_aux]).shape == (2, 2)
    assert eqmarl.name != qfctde.name
