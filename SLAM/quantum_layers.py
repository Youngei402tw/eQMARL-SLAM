"""Self-contained TensorFlow Quantum layers used by the SLAM critics.

The implementation follows the hybrid variational data-reuploading pattern of
the eQMARL reference code while removing the missing Git-submodule dependency.
Quantum imports are optional so the classical baselines remain usable without
Cirq or TensorFlow Quantum.
"""
from __future__ import annotations

import functools
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

try:
    import cirq
    import sympy
    import tensorflow_quantum as tfq
    _QUANTUM_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - depends on optional environment.
    cirq = None  # type: ignore[assignment]
    sympy = None  # type: ignore[assignment]
    tfq = None  # type: ignore[assignment]
    _QUANTUM_IMPORT_ERROR = exc


def quantum_dependencies_available() -> bool:
    return _QUANTUM_IMPORT_ERROR is None


def require_quantum_dependencies() -> None:
    if _QUANTUM_IMPORT_ERROR is not None:
        raise ImportError(
            "quantum frameworks require cirq, sympy, and tensorflow-quantum; "
            "install requirements-quantum.txt or run only sctde/fctde"
        ) from _QUANTUM_IMPORT_ERROR


def _pauli_product(qubits: Sequence["cirq.Qid"]):
    return functools.reduce(
        lambda left, right: left * right,
        (cirq.Z(qubit) for qubit in qubits),
    )


def make_readout_observables(
    qubits: Sequence["cirq.Qid"],
    n_agents: int,
    d_qubits: int,
) -> List["cirq.PauliString"]:
    """Per-agent partition Z, X, Y observables for richer quantum readout."""
    require_quantum_dependencies()
    observables: List["cirq.PauliString"] = []

    def append_unique(observable) -> None:
        if observable not in observables:
            observables.append(observable)

    for agent in range(n_agents):
        partition = qubits[agent * d_qubits:(agent + 1) * d_qubits]
        append_unique(_pauli_product(partition))
    for agent in range(n_agents):
        partition = qubits[agent * d_qubits:(agent + 1) * d_qubits]
        pauli_x_product = functools.reduce(
            lambda left, right: left * right,
            (cirq.X(qubit) for qubit in partition),
        )
        append_unique(pauli_x_product)
    for agent in range(n_agents):
        partition = qubits[agent * d_qubits:(agent + 1) * d_qubits]
        pauli_y_product = functools.reduce(
            lambda left, right: left * right,
            (cirq.Y(qubit) for qubit in partition),
        )
        append_unique(pauli_y_product)
    return observables


def _apply_initial_entanglement(
    circuit: "cirq.Circuit",
    qubits: Sequence["cirq.Qid"],
    n_agents: int,
    d_qubits: int,
    entanglement_type: str,
) -> None:
    entanglement_type = entanglement_type.lower()
    if entanglement_type not in {"phi+", "phi-", "psi+", "psi-"}:
        raise ValueError("entanglement_type must be phi+, phi-, psi+, or psi-")
    for local_index in range(d_qubits):
        control = qubits[local_index]
        circuit.append(cirq.H(control))
        for agent in range(1, n_agents):
            target = qubits[agent * d_qubits + local_index]
            if entanglement_type.startswith("psi"):
                circuit.append(cirq.X(target))
            circuit.append(cirq.CNOT(control, target))
        if entanglement_type.endswith("-"):
            circuit.append(cirq.Z(control))


def _append_rotation_triplet(
    circuit: "cirq.Circuit",
    qubit: "cirq.Qid",
    symbols,
) -> None:
    rx, ry, rz = symbols
    circuit.append((cirq.rx(rx)(qubit), cirq.ry(ry)(qubit), cirq.rz(rz)(qubit)))


def _append_ring_cz(
    circuit: "cirq.Circuit",
    qubits: Sequence["cirq.Qid"],
) -> None:
    if len(qubits) <= 1:
        return
    for left, right in zip(qubits, qubits[1:]):
        circuit.append(cirq.CZ(left, right))
    if len(qubits) > 2:
        circuit.append(cirq.CZ(qubits[-1], qubits[0]))


def build_partite_variational_circuit(
    n_agents: int,
    d_qubits: int,
    n_layers: int,
    input_entanglement: bool,
    entanglement_type: str = "psi+",
) -> Tuple[Sequence["cirq.Qid"], "cirq.Circuit", np.ndarray, np.ndarray]:
    """Build split local circuits linked by optional initial entanglement."""
    require_quantum_dependencies()
    if n_agents < 1 or d_qubits < 1 or n_layers < 1:
        raise ValueError("n_agents, d_qubits, and n_layers must be positive")
    n_qubits = n_agents * d_qubits
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    if input_entanglement and n_agents > 1:
        _apply_initial_entanglement(
            circuit,
            qubits,
            n_agents=n_agents,
            d_qubits=d_qubits,
            entanglement_type=entanglement_type,
        )

    variational_symbols = np.asarray(
        sympy.symbols(f"theta0:{n_layers * n_agents * d_qubits * 3}"), dtype=object
    ).reshape(n_layers, n_agents, d_qubits, 3)
    encoding_symbols = np.asarray(
        sympy.symbols(f"input0:{n_agents * n_layers * d_qubits * 3}"),
        dtype=object,
    ).reshape(n_agents, n_layers, d_qubits, 3)

    for layer in range(n_layers):
        for agent in range(n_agents):
            for local_index in range(d_qubits):
                qubit = qubits[agent * d_qubits + local_index]
                _append_rotation_triplet(
                    circuit,
                    qubit,
                    encoding_symbols[agent, layer, local_index],
                )
        for agent in range(n_agents):
            for local_index in range(d_qubits):
                qubit = qubits[agent * d_qubits + local_index]
                _append_rotation_triplet(
                    circuit,
                    qubit,
                    variational_symbols[layer, agent, local_index],
                )
        for agent in range(n_agents):
            partition = qubits[agent * d_qubits:(agent + 1) * d_qubits]
            _append_ring_cz(circuit, partition)
        if n_agents > 1:
            for agent in range(n_agents):
                last_qubit = qubits[agent * d_qubits + d_qubits - 1]
                next_agent = (agent + 1) % n_agents
                first_qubit = qubits[next_agent * d_qubits]
                circuit.append(cirq.CZ(last_qubit, first_qubit))
    return qubits, circuit, variational_symbols, encoding_symbols


def build_central_variational_circuit(
    n_agents: int,
    d_qubits: int,
    n_layers: int,
) -> Tuple[Sequence["cirq.Qid"], "cirq.Circuit", np.ndarray, np.ndarray]:
    """Build one fully centralized circuit over all agent qubits."""
    require_quantum_dependencies()
    if n_agents < 1 or d_qubits < 1 or n_layers < 1:
        raise ValueError("n_agents, d_qubits, and n_layers must be positive")
    n_qubits = n_agents * d_qubits
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    variational_symbols = np.asarray(
        sympy.symbols(f"theta0:{n_layers * n_qubits * 3}"), dtype=object
    ).reshape(n_layers, n_qubits, 3)
    encoding_symbols = np.asarray(
        sympy.symbols(f"input0:{n_layers * n_qubits * 3}"), dtype=object
    ).reshape(n_layers, n_qubits, 3)

    for layer in range(n_layers):
        for qubit_index, qubit in enumerate(qubits):
            _append_rotation_triplet(
                circuit,
                qubit,
                encoding_symbols[layer, qubit_index],
            )
        for qubit_index, qubit in enumerate(qubits):
            _append_rotation_triplet(
                circuit,
                qubit,
                variational_symbols[layer, qubit_index],
            )
        _append_ring_cz(circuit, qubits)
    return qubits, circuit, variational_symbols, encoding_symbols


class _BaseVariationalPQC(keras.layers.Layer):
    """Shared batching, trainable weights, and TFQ symbol ordering."""

    def __init__(
        self,
        n_agents: int,
        d_qubits: int,
        n_layers: int,
        squash_activation: str = "atan",
        trainable_encoding_scale: bool = True,
        **kwargs,
    ) -> None:
        require_quantum_dependencies()
        for name, value in (
            ("n_agents", n_agents),
            ("d_qubits", d_qubits),
            ("n_layers", n_layers),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        super().__init__(**kwargs)
        self.n_agents = int(n_agents)
        self.d_qubits = int(d_qubits)
        self.n_layers = int(n_layers)
        self.squash_activation = str(squash_activation)
        self.trainable_encoding_scale = bool(trainable_encoding_scale)

    def _configure_quantum_objects(
        self,
        qubits,
        circuit,
        variational_symbols: np.ndarray,
        encoding_symbols: np.ndarray,
    ) -> None:
        self.qubits = tuple(qubits)
        self.circuit = circuit
        self.observables = tuple(
            make_readout_observables(
                qubits,
                n_agents=self.n_agents,
                d_qubits=self.d_qubits,
            )
        )
        self._variational_shape = tuple(variational_symbols.shape)
        self._encoding_shape = tuple(encoding_symbols.shape)

        symbols = [
            str(symbol)
            for symbol in np.concatenate(
                (variational_symbols.reshape(-1), encoding_symbols.reshape(-1))
            )
        ]
        self._symbol_order_indices = tf.constant(
            [symbols.index(symbol) for symbol in sorted(symbols)],
            dtype=tf.int32,
        )
        self._empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self._controlled_pqc = tfq.layers.ControlledPQC(
            circuit,
            list(self.observables),
        )

    def build(self, input_shape) -> None:
        self.variational_weights = self.add_weight(
            name="variational_weights",
            shape=self._variational_shape,
            initializer=keras.initializers.RandomUniform(
                minval=0.0,
                maxval=np.pi,
            ),
            trainable=True,
            dtype=tf.float32,
        )
        self.encoding_scales = self.add_weight(
            name="encoding_scales",
            shape=self._encoding_shape,
            initializer="ones",
            trainable=self.trainable_encoding_scale,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def _squash(self, tensor):
        if self.squash_activation in {"atan", "arctan"}:
            return tf.math.atan(tensor)
        if self.squash_activation == "tanh":
            return tf.math.tanh(tensor)
        if self.squash_activation in {"linear", "none"}:
            return tensor
        return keras.activations.get(self.squash_activation)(tensor)

    def _execute(self, encoding_angles):
        batch_size = tf.gather(tf.shape(encoding_angles), 0)
        circuits = tf.repeat(self._empty_circuit, repeats=batch_size)
        multiples = [batch_size] + [1] * len(self._variational_shape)
        tiled_variational = tf.tile(
            self.variational_weights[tf.newaxis, ...],
            multiples=multiples,
        )
        variational_angles = tf.reshape(tiled_variational, [batch_size, -1])
        joined_angles = tf.concat(
            [variational_angles, tf.reshape(encoding_angles, [batch_size, -1])],
            axis=1,
        )
        ordered_angles = tf.gather(
            joined_angles,
            self._symbol_order_indices,
            axis=1,
        )
        return self._controlled_pqc([circuits, ordered_angles])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.observables))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_agents": self.n_agents,
                "d_qubits": self.d_qubits,
                "n_layers": self.n_layers,
                "squash_activation": self.squash_activation,
                "trainable_encoding_scale": self.trainable_encoding_scale,
            }
        )
        return config


class PartiteVariationalPQC(_BaseVariationalPQC):
    """Split eQMARL PQC with independent local partitions."""

    def __init__(
        self,
        n_agents: int,
        d_qubits: int,
        n_layers: int,
        input_entanglement: bool = True,
        entanglement_type: str = "psi+",
        **kwargs,
    ) -> None:
        self.input_entanglement = bool(input_entanglement)
        self.entanglement_type = str(entanglement_type)
        super().__init__(
            n_agents=n_agents,
            d_qubits=d_qubits,
            n_layers=n_layers,
            **kwargs,
        )
        objects = build_partite_variational_circuit(
            n_agents=self.n_agents,
            d_qubits=self.d_qubits,
            n_layers=self.n_layers,
            input_entanglement=self.input_entanglement,
            entanglement_type=self.entanglement_type,
        )
        self._configure_quantum_objects(*objects)

    def build(self, input_shape) -> None:
        expected = (self.n_agents, self.d_qubits, 3)
        actual = tuple(
            None if dimension is None else int(dimension)
            for dimension in input_shape[-3:]
        )
        if actual != expected:
            raise ValueError(
                f"PartiteVariationalPQC expects input suffix {expected}, "
                f"got {actual}"
            )
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        # w_enc[p,l,q,f] * input[b,p,q,f] -> b,p,l,q,f
        angles = tf.einsum(
            "plqf,bpqf->bplqf",
            self.encoding_scales,
            inputs,
        )
        return self._execute(self._squash(angles))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_entanglement": self.input_entanglement,
                "entanglement_type": self.entanglement_type,
            }
        )
        return config


class CentralVariationalPQC(_BaseVariationalPQC):
    """Fully centralized PQC with variational entanglement over all qubits."""

    def __init__(
        self,
        n_agents: int,
        d_qubits: int,
        n_layers: int,
        **kwargs,
    ) -> None:
        super().__init__(
            n_agents=n_agents,
            d_qubits=d_qubits,
            n_layers=n_layers,
            **kwargs,
        )
        objects = build_central_variational_circuit(
            n_agents=self.n_agents,
            d_qubits=self.d_qubits,
            n_layers=self.n_layers,
        )
        self._configure_quantum_objects(*objects)

    def build(self, input_shape) -> None:
        expected = (self.n_agents * self.d_qubits, 3)
        actual = tuple(
            None if dimension is None else int(dimension)
            for dimension in input_shape[-2:]
        )
        if actual != expected:
            raise ValueError(
                f"CentralVariationalPQC expects input suffix {expected}, "
                f"got {actual}"
            )
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        # w_enc[l,q,f] * input[b,q,f] -> b,l,q,f
        angles = tf.einsum(
            "lqf,bqf->blqf",
            self.encoding_scales,
            inputs,
        )
        return self._execute(self._squash(angles))


__all__ = [
    "CentralVariationalPQC",
    "PartiteVariationalPQC",
    "build_central_variational_circuit",
    "build_partite_variational_circuit",
    "make_readout_observables",
    "quantum_dependencies_available",
    "require_quantum_dependencies",
]
