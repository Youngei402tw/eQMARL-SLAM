"""Corrected eQMARL-SLAM package.

The pure NumPy simulator is imported eagerly.  Gymnasium and TensorFlow modules
remain opt-in so map-generation tests can run without the learning stack.
"""
from __future__ import annotations

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

__version__ = "2.0.0"


def register_environments() -> None:
    """Import the Gymnasium interface and register the named environments."""
    from . import slam as _slam  # noqa: F401


__all__ = [
    "ACTION_FORWARD",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "AUX_DIM",
    "N_ACTIONS",
    "OBS_CHANNELS",
    "OBSERVATION_CHANNELS",
    "CooperativeSLAMCore",
    "RewardConfig",
    "register_environments",
    "__version__",
]
