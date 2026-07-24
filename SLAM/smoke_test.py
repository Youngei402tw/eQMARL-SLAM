#!/usr/bin/env python3
"""Dependency-aware smoke test for the corrected repository."""
from __future__ import annotations

import importlib.util
import sys

import numpy as np

try:
    from .slam_core import ACTION_FORWARD, ACTION_LEFT, CooperativeSLAMCore
except ImportError:
    from slam_core import ACTION_FORWARD, ACTION_LEFT, CooperativeSLAMCore


def main() -> int:
    core = CooperativeSLAMCore(size=8, maze_type="random", max_steps=5)
    observation, _ = core.reset(seed=2026)
    assert observation["image"].shape == (2, 8, 8, 5)
    for _ in range(3):
        observation, reward, terminated, truncated, info = core.step(
            [ACTION_FORWARD, ACTION_LEFT]
        )
        assert np.all(np.isfinite(reward))
        assert 0.0 <= info["metrics"]["coverage"] <= 1.0
        if terminated or truncated:
            break
    print("[ok] pure NumPy SLAM core")

    if importlib.util.find_spec("gymnasium") is None:
        print("[skip] Gymnasium interface (dependency unavailable)")
    else:
        try:
            from .slam import MultiAgentSLAMEnv
        except ImportError:
            from slam import MultiAgentSLAMEnv
        env = MultiAgentSLAMEnv(size=8, max_steps=5)
        obs, _ = env.reset(seed=2026)
        assert env.observation_space.contains(obs)
        env.close()
        print("[ok] Gymnasium interface")

    if importlib.util.find_spec("tensorflow") is None:
        print("[skip] classical models/MAA2C (TensorFlow unavailable)")
    else:
        try:
            from .slam_baselines import generate_actor, generate_critic_sctde
        except ImportError:
            from slam_baselines import generate_actor, generate_critic_sctde
        actor = generate_actor(obs_shape=(8, 8, 5))
        critic = generate_critic_sctde(n_agents=2, obs_shape=(8, 8, 5))
        assert actor.output_shape == (None, 3)
        assert critic.output_shape == (None, 2)
        print("[ok] TensorFlow actor and classical critic")

    if importlib.util.find_spec("tensorflow_quantum") is None:
        print("[skip] quantum critics (TensorFlow Quantum unavailable)")
    else:
        try:
            from .slam_baselines import generate_critic_eqmarl_psi_plus
        except ImportError:
            from slam_baselines import generate_critic_eqmarl_psi_plus
        critic = generate_critic_eqmarl_psi_plus(
            n_agents=2,
            d_qubits=2,
            n_layers=1,
            obs_shape=(6, 6, 5),
        )
        assert critic.output_shape == (None, 2)
        print("[ok] TensorFlow Quantum critic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
