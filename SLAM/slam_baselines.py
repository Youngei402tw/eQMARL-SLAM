"""Actor and four distinct critic architectures for the SLAM benchmark."""
from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

try:
    from .slam_core import AUX_DIM, N_ACTIONS, OBS_CHANNELS
except ImportError:  # Support direct execution from SLAM/.
    from slam_core import AUX_DIM, N_ACTIONS, OBS_CHANNELS

FrameworkBuilder = Callable[..., keras.Model]

FRAMEWORK_DESCRIPTIONS = {
    "eqmarl": (
        "split per-agent classical encoders feeding local quantum partitions "
        "with psi+ entanglement, per-agent variational weights, and "
        "learnable inter-agent entanglement in variational layers"
    ),
    "qfctde": (
        "fully centralized classical preprocessor feeding one quantum circuit "
        "with variational entanglement over all qubits"
    ),
    "sctde": (
        "independent per-agent classical branches aggregated only after local encoding"
    ),
    "fctde": "one fully centralized classical network over the raw joint observation",
}


def _local_spatial_features(
    image_tensor,
    prefix: str,
    filters: Sequence[int] = (16, 32),
    dense_units: int = 64,
):
    features = image_tensor
    for layer_index, filter_count in enumerate(filters):
        features = keras.layers.Conv2D(
            int(filter_count),
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name=f"{prefix}-conv-{layer_index}",
        )(features)
    features = keras.layers.Flatten(name=f"{prefix}-flatten")(features)
    return keras.layers.Dense(
        int(dense_units),
        activation="relu",
        name=f"{prefix}-dense",
    )(features)


def generate_actor(
    n_actions: int = N_ACTIONS,
    obs_shape: Tuple[int, int, int] = (8, 8, OBS_CHANNELS),
    aux_dim: int = AUX_DIM,
    hidden_units: int = 64,
    **kwargs,
) -> keras.Model:
    """Shared decentralized actor over one agent's local state."""
    image_input = keras.Input(shape=obs_shape, dtype=tf.float32, name="image")
    aux_input = keras.Input(shape=(aux_dim,), dtype=tf.float32, name="aux")
    spatial = _local_spatial_features(
        image_input,
        prefix="actor-local",
        filters=(16, 32),
        dense_units=hidden_units,
    )
    state = keras.layers.Concatenate(name="actor-state")([spatial, aux_input])
    state = keras.layers.Dense(
        int(hidden_units),
        activation="relu",
        name="actor-hidden",
    )(state)
    probabilities = keras.layers.Dense(
        int(n_actions),
        activation="softmax",
        name="policy",
    )(state)
    return keras.Model(
        inputs=[image_input, aux_input],
        outputs=probabilities,
        name="actor",
        **kwargs,
    )


def _joint_inputs(
    n_agents: int,
    obs_shape: Tuple[int, int, int],
    aux_dim: int,
):
    image_input = keras.Input(
        shape=(n_agents, *obs_shape),
        dtype=tf.float32,
        name="joint-image",
    )
    aux_input = keras.Input(
        shape=(n_agents, aux_dim),
        dtype=tf.float32,
        name="joint-aux",
    )
    return image_input, aux_input


def _slice_agent(tensor, agent_index: int, output_shape, name: str):
    return keras.layers.Lambda(
        lambda value, index=agent_index: value[:, index],
        output_shape=output_shape,
        name=name,
    )(tensor)


def generate_critic_eqmarl_psi_plus(
    n_agents: int = 2,
    d_qubits: int = 4,
    n_layers: int = 5,
    obs_shape: Tuple[int, int, int] = (8, 8, OBS_CHANNELS),
    aux_dim: int = AUX_DIM,
    local_hidden: int = 48,
    squash_activation: str = "atan",
    **kwargs,
) -> keras.Model:
    """eQMARL critic with local encoders and psi+ entangled partitions.

    ``local_hidden`` is intentionally smaller than the sCTDE ``local_hidden``
    (48 < 64) so that the overall parameter count stays well below the
    classical baselines while the quantum PQC provides additional expressive
    capacity through entangled observables.
    """
    try:
        from .quantum_layers import (
            PartiteVariationalPQC,
            require_quantum_dependencies,
        )
    except ImportError:
        from quantum_layers import PartiteVariationalPQC, require_quantum_dependencies

    require_quantum_dependencies()
    image_input, aux_input = _joint_inputs(n_agents, obs_shape, aux_dim)
    agent_angles = []
    for agent_index in range(n_agents):
        local_image = _slice_agent(
            image_input,
            agent_index,
            obs_shape,
            name=f"eqmarl-agent-{agent_index}-image",
        )
        local_aux = _slice_agent(
            aux_input,
            agent_index,
            (aux_dim,),
            name=f"eqmarl-agent-{agent_index}-aux",
        )
        local_state = _local_spatial_features(
            local_image,
            prefix=f"eqmarl-agent-{agent_index}",
            filters=(8, 16),
            dense_units=local_hidden,
        )
        local_state = keras.layers.Concatenate(
            name=f"eqmarl-agent-{agent_index}-state"
        )([local_state, local_aux])
        angles = keras.layers.Dense(
            d_qubits * 3,
            activation="tanh",
            name=f"eqmarl-agent-{agent_index}-angles",
        )(local_state)
        angles = keras.layers.Reshape(
            (d_qubits, 3),
            name=f"eqmarl-agent-{agent_index}-angle-grid",
        )(angles)
        agent_angles.append(angles)

    partitions = keras.layers.Lambda(
        lambda tensors: tf.stack(tensors, axis=1),
        output_shape=(n_agents, d_qubits, 3),
        name="eqmarl-partitions",
    )(agent_angles)
    expectations = PartiteVariationalPQC(
        n_agents=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        input_entanglement=True,
        entanglement_type="psi+",
        squash_activation=squash_activation,
        name="eqmarl-pqc",
    )(partitions)
    readout = keras.layers.Dense(
        8,
        activation="tanh",
        name="eqmarl-readout-hidden",
    )(expectations)
    readout = keras.layers.Dense(
        4,
        activation="tanh",
        name="eqmarl-readout-2",
    )(readout)
    value = keras.layers.Dense(n_agents, activation=None, name="v")(readout)
    return keras.Model(
        inputs=[image_input, aux_input],
        outputs=value,
        name="critic-eqmarl-psi-plus",
        **kwargs,
    )


def generate_critic_qfctde(
    n_agents: int = 2,
    d_qubits: int = 4,
    n_layers: int = 5,
    obs_shape: Tuple[int, int, int] = (8, 8, OBS_CHANNELS),
    aux_dim: int = AUX_DIM,
    centralized_hidden: int = 128,
    squash_activation: str = "atan",
    **kwargs,
) -> keras.Model:
    """Fully centralized quantum CTDE critic."""
    try:
        from .quantum_layers import CentralVariationalPQC, require_quantum_dependencies
    except ImportError:
        from quantum_layers import CentralVariationalPQC, require_quantum_dependencies

    require_quantum_dependencies()
    image_input, aux_input = _joint_inputs(n_agents, obs_shape, aux_dim)
    flat_image = keras.layers.Flatten(name="qfctde-flat-image")(image_input)
    flat_aux = keras.layers.Flatten(name="qfctde-flat-aux")(aux_input)
    joint_state = keras.layers.Concatenate(name="qfctde-joint-state")(
        [flat_image, flat_aux]
    )
    joint_state = keras.layers.Dense(
        int(centralized_hidden),
        activation="relu",
        name="qfctde-central-hidden",
    )(joint_state)
    angles = keras.layers.Dense(
        n_agents * d_qubits * 3,
        activation="tanh",
        name="qfctde-central-angles",
    )(joint_state)
    angles = keras.layers.Reshape(
        (n_agents * d_qubits, 3),
        name="qfctde-angle-grid",
    )(angles)
    expectations = CentralVariationalPQC(
        n_agents=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        squash_activation=squash_activation,
        name="qfctde-pqc",
    )(angles)
    readout = keras.layers.Dense(
        8,
        activation="tanh",
        name="qfctde-readout-hidden",
    )(expectations)
    value = keras.layers.Dense(n_agents, activation=None, name="v")(readout)
    return keras.Model(
        inputs=[image_input, aux_input],
        outputs=value,
        name="critic-qfctde",
        **kwargs,
    )


def generate_critic_sctde(
    n_agents: int = 2,
    obs_shape: Tuple[int, int, int] = (8, 8, OBS_CHANNELS),
    aux_dim: int = AUX_DIM,
    local_hidden: int = 64,
    joint_hidden: int = 64,
    **kwargs,
) -> keras.Model:
    """Split classical CTDE critic with independent local branches."""
    image_input, aux_input = _joint_inputs(n_agents, obs_shape, aux_dim)
    local_latents = []
    for agent_index in range(n_agents):
        local_image = _slice_agent(
            image_input,
            agent_index,
            obs_shape,
            name=f"sctde-agent-{agent_index}-image",
        )
        local_aux = _slice_agent(
            aux_input,
            agent_index,
            (aux_dim,),
            name=f"sctde-agent-{agent_index}-aux",
        )
        local_state = _local_spatial_features(
            local_image,
            prefix=f"sctde-agent-{agent_index}",
            filters=(8, 16),
            dense_units=local_hidden,
        )
        local_state = keras.layers.Concatenate(
            name=f"sctde-agent-{agent_index}-state"
        )([local_state, local_aux])
        local_latent = keras.layers.Dense(
            int(local_hidden),
            activation="relu",
            name=f"sctde-agent-{agent_index}-latent",
        )(local_state)
        local_latents.append(local_latent)

    joint_state = keras.layers.Concatenate(name="sctde-latent-aggregation")(
        local_latents
    )
    joint_state = keras.layers.Dense(
        int(joint_hidden),
        activation="relu",
        name="sctde-joint-hidden",
    )(joint_state)
    value = keras.layers.Dense(n_agents, activation=None, name="v")(joint_state)
    return keras.Model(
        inputs=[image_input, aux_input],
        outputs=value,
        name="critic-sctde",
        **kwargs,
    )


def generate_critic_fctde(
    n_agents: int = 2,
    obs_shape: Tuple[int, int, int] = (8, 8, OBS_CHANNELS),
    aux_dim: int = AUX_DIM,
    hidden_units: Sequence[int] = (128, 64),
    **kwargs,
) -> keras.Model:
    """Fully centralized classical critic over the raw joint state."""
    image_input, aux_input = _joint_inputs(n_agents, obs_shape, aux_dim)
    flat_image = keras.layers.Flatten(name="fctde-flat-image")(image_input)
    flat_aux = keras.layers.Flatten(name="fctde-flat-aux")(aux_input)
    joint_state = keras.layers.Concatenate(name="fctde-joint-state")(
        [flat_image, flat_aux]
    )
    for layer_index, units in enumerate(hidden_units):
        joint_state = keras.layers.Dense(
            int(units),
            activation="relu",
            name=f"fctde-hidden-{layer_index}",
        )(joint_state)
    value = keras.layers.Dense(n_agents, activation=None, name="v")(joint_state)
    return keras.Model(
        inputs=[image_input, aux_input],
        outputs=value,
        name="critic-fctde",
        **kwargs,
    )


def framework_requires_quantum(framework: str) -> bool:
    return framework.lower() in {"eqmarl", "qfctde"}


def get_critic_builder(framework: str) -> FrameworkBuilder:
    builders: Dict[str, FrameworkBuilder] = {
        "eqmarl": generate_critic_eqmarl_psi_plus,
        "qfctde": generate_critic_qfctde,
        "sctde": generate_critic_sctde,
        "fctde": generate_critic_fctde,
    }
    key = framework.lower()
    if key not in builders:
        raise ValueError(
            f"unknown framework {framework!r}; choose from {sorted(builders)}"
        )
    return builders[key]


def get_optimizer_configs(
    framework: str,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-3,
    quantum_critic_lr: float = 1e-3,
):
    actor_optimizer = keras.optimizers.Adam(learning_rate=float(actor_lr))
    selected_critic_lr = (
        quantum_critic_lr if framework_requires_quantum(framework) else critic_lr
    )
    critic_optimizer = keras.optimizers.Adam(
        learning_rate=float(selected_critic_lr)
    )
    return actor_optimizer, critic_optimizer


def count_trainable_params(model: keras.Model) -> int:
    """Return a backend-agnostic trainable parameter count."""
    return int(
        sum(np.prod(tuple(int(dim) for dim in variable.shape))
            for variable in model.trainable_variables)
    )


__all__ = [
    "FRAMEWORK_DESCRIPTIONS",
    "count_trainable_params",
    "framework_requires_quantum",
    "generate_actor",
    "generate_critic_eqmarl_psi_plus",
    "generate_critic_fctde",
    "generate_critic_qfctde",
    "generate_critic_sctde",
    "get_critic_builder",
    "get_optimizer_configs",
]
