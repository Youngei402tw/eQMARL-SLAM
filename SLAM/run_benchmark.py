#!/usr/bin/env python3
"""Train and evaluate eQMARL, qfCTDE, sCTDE, and fCTDE fairly."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# TensorFlow 2.7 uses environment flags for deterministic GPU kernels.
# setdefault preserves an explicit user override.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

try:
    from .maa2c import MAA2C
    from .quantum_layers import quantum_dependencies_available
    from .slam import make_multi_agent_env_from_id, parse_env_id
    from .slam_baselines import (
        FRAMEWORK_DESCRIPTIONS,
        count_trainable_params,
        framework_requires_quantum,
        generate_actor,
        get_critic_builder,
        get_optimizer_configs,
    )
    from .slam_core import AUX_DIM, OBS_CHANNELS, OBSERVATION_CHANNELS, RewardConfig
except ImportError:  # Support ``cd SLAM && python run_benchmark.py``.
    from maa2c import MAA2C
    from quantum_layers import quantum_dependencies_available
    from slam import make_multi_agent_env_from_id, parse_env_id
    from slam_baselines import (
        FRAMEWORK_DESCRIPTIONS,
        count_trainable_params,
        framework_requires_quantum,
        generate_actor,
        get_critic_builder,
        get_optimizer_configs,
    )
    from slam_core import AUX_DIM, OBS_CHANNELS, OBSERVATION_CHANNELS, RewardConfig

DEFAULT_MAP_CONFIGS = {
    6: {"max_steps": 50, "episodes": 500},
    8: {"max_steps": 100, "episodes": 2000},
    16: {"max_steps": 150, "episodes": 1000},
    32: {"max_steps": 200, "episodes": 1000},
}
ALL_FRAMEWORKS = ("eqmarl", "qfctde", "sctde", "fctde")
SCRIPT_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _environment_frameworks() -> List[str]:
    raw = os.environ.get("SLAM_FRAMEWORKS", ",".join(ALL_FRAMEWORKS))
    return [item.strip().lower() for item in raw.replace(",", " ").split() if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paired-seed cooperative SLAM benchmark with held-out evaluation. "
            "Quantum frameworks are skipped when TFQ is unavailable unless "
            "--strict-quantum is set."
        )
    )
    parser.add_argument(
        "--env-id",
        default=os.environ.get("SLAM_ENV_ID", "MiniGrid-RandomSLAM-8x8-v0"),
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=_environment_frameworks(),
        help="space-separated subset of: eqmarl qfctde sctde fctde",
    )
    parser.add_argument("--n-agents", type=int, default=_env_int("SLAM_N_AGENTS", 2))
    parser.add_argument("--n-seeds", type=int, default=_env_int("SLAM_N_SEEDS", 5))
    parser.add_argument(
        "--episodes",
        "--n-episodes",
        dest="episodes",
        type=int,
        default=(
            int(os.environ["SLAM_N_EPISODES"])
            if "SLAM_N_EPISODES" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=(
            int(os.environ["SLAM_MAX_STEPS"])
            if "SLAM_MAX_STEPS" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=_env_int("SLAM_EVAL_EPISODES", 100),
    )
    parser.add_argument("--quick", action="store_true", help="20-episode smoke run")

    parser.add_argument("--model-seed-start", type=int, default=0)
    parser.add_argument("--train-map-seed", type=int, default=17_291)
    parser.add_argument("--train-action-seed", type=int, default=48_211)
    parser.add_argument("--eval-map-seed", type=int, default=91_733)
    parser.add_argument("--eval-action-seed", type=int, default=63_017)

    parser.add_argument(
        "--obstacle-density",
        type=float,
        default=_env_float("SLAM_OBSTACLE_DENSITY", 0.15),
    )
    parser.add_argument("--fov-range", type=int, default=4)
    parser.add_argument("--fov-angle-degrees", type=float, default=90.0)
    parser.add_argument("--target-coverage", type=float, default=0.95)

    parser.add_argument("--information-gain-scale", type=float, default=10.0)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--collision-penalty", type=float, default=0.05)
    parser.add_argument("--redundancy-penalty", type=float, default=0.0)
    parser.add_argument("--completion-bonus", type=float, default=1.0)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--quantum-critic-lr", type=float, default=1e-3)
    parser.add_argument("--d-qubits", type=int, default=4)
    parser.add_argument("--quantum-layers", type=int, default=5)

    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="sample evaluation actions instead of using argmax",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--strict-quantum", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SLAM_OUTPUT_DIR", SCRIPT_DIR / "results")),
    )
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    maze_type, size = parse_env_id(args.env_id)
    del maze_type
    if size not in DEFAULT_MAP_CONFIGS:
        raise ValueError(f"no default training configuration for map size {size}")
    defaults = DEFAULT_MAP_CONFIGS[size]
    if args.episodes is None:
        args.episodes = defaults["episodes"]
    if args.max_steps is None:
        args.max_steps = defaults["max_steps"]
    if args.quick:
        args.n_seeds = 1
        args.episodes = 20
        args.max_steps = min(int(args.max_steps), 30)
        args.eval_episodes = min(int(args.eval_episodes), 10)

    args.frameworks = list(dict.fromkeys(item.lower() for item in args.frameworks))
    unknown = sorted(set(args.frameworks) - set(ALL_FRAMEWORKS))
    if unknown:
        raise ValueError(f"unknown frameworks: {unknown}")
    if not args.frameworks:
        raise ValueError("at least one framework must be selected")
    for field in ("n_agents", "n_seeds", "episodes", "max_steps", "eval_episodes"):
        if int(getattr(args, field)) < 1:
            raise ValueError(f"{field} must be positive")
    if args.d_qubits < 1 or args.quantum_layers < 1:
        raise ValueError("d_qubits and quantum_layers must be positive")
    for seed_field in (
        "model_seed_start",
        "train_map_seed",
        "train_action_seed",
        "eval_map_seed",
        "eval_action_seed",
    ):
        if int(getattr(args, seed_field)) < 0:
            raise ValueError(f"{seed_field} must be non-negative")
    if not 0.0 <= args.obstacle_density < 0.5:
        raise ValueError("obstacle_density must be in [0, 0.5)")
    if args.fov_range < 1:
        raise ValueError("fov_range must be positive")
    if not 0.0 < args.fov_angle_degrees <= 360.0:
        raise ValueError("fov_angle_degrees must be in (0, 360]")
    if not 0.0 < args.target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    for reward_field in (
        "information_gain_scale",
        "step_penalty",
        "collision_penalty",
        "redundancy_penalty",
        "completion_bonus",
        "entropy_coef",
    ):
        value = float(getattr(args, reward_field))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{reward_field} must be finite and non-negative")
    for bounded_field in ("gamma", "gae_lambda"):
        value = float(getattr(args, bounded_field))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{bounded_field} must be finite and in [0, 1]")
    for positive_field in (
        "max_grad_norm",
        "actor_lr",
        "critic_lr",
        "quantum_critic_lr",
    ):
        value = float(getattr(args, positive_field))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{positive_field} must be finite and positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def configure_tensorflow() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    enable_determinism = getattr(
        getattr(tf.config, "experimental", object()),
        "enable_op_determinism",
        None,
    )
    if callable(enable_determinism):
        try:
            enable_determinism()
        except Exception:
            # Older accelerators/TF builds can reject deterministic kernels.
            pass


def seed_everything(seed: int) -> None:
    # PYTHONHASHSEED is fixed at interpreter startup; runtime assignment would
    # be misleading, so only live RNGs are seeded here.
    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


def make_seed_schedule(base_seed: int, model_seed: Optional[int], count: int) -> List[int]:
    if isinstance(base_seed, bool) or int(base_seed) != base_seed or int(base_seed) < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if model_seed is not None and (
        isinstance(model_seed, bool)
        or int(model_seed) != model_seed
        or int(model_seed) < 0
    ):
        raise ValueError("model_seed must be a non-negative integer or None")
    if isinstance(count, bool) or int(count) != count or int(count) < 0:
        raise ValueError("count must be a non-negative integer")
    entropy = [int(base_seed)] if model_seed is None else [int(base_seed), int(model_seed)]
    sequence = np.random.SeedSequence(entropy)
    selected: List[int] = []
    seen = set()
    while len(selected) < int(count):
        children = sequence.spawn(max(32, int(count) - len(selected)))
        for child in children:
            candidate = int(child.generate_state(1, dtype=np.uint32)[0])
            if candidate in seen:
                continue
            seen.add(candidate)
            selected.append(candidate)
            if len(selected) == int(count):
                break
    return selected


def make_disjoint_seed_schedule(
    base_seed: int,
    count: int,
    forbidden: Iterable[int],
) -> List[int]:
    """Generate deterministic episode seeds excluding a supplied seed set."""
    forbidden_set = {int(value) for value in forbidden}
    selected: List[int] = []
    offset = 0
    while len(selected) < count:
        candidates = make_seed_schedule(
            base_seed=int(base_seed) + offset,
            model_seed=None,
            count=max(count * 2, 32),
        )
        for candidate in candidates:
            if candidate not in forbidden_set and candidate not in selected:
                selected.append(candidate)
                if len(selected) == count:
                    break
        offset += 1
    return selected


def serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize object of type {type(value).__name__}")


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=serializable,
            allow_nan=False,
        )
    os.replace(temporary, path)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_weight_pair_atomically(
    actor: keras.Model,
    critic: keras.Model,
    actor_path: Path,
    critic_path: Path,
) -> None:
    """Save both HDF5 weight files without exposing partial temporary files."""
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    critic_path.parent.mkdir(parents=True, exist_ok=True)
    actor_temporary = actor_path.with_name(
        actor_path.name.replace(".weights.h5", ".tmp.weights.h5")
    )
    critic_temporary = critic_path.with_name(
        critic_path.name.replace(".weights.h5", ".tmp.weights.h5")
    )
    for final_path in (actor_path, critic_path):
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite existing weights: {final_path}")
    for temporary in (actor_temporary, critic_temporary):
        temporary.unlink(missing_ok=True)
    committed: List[Path] = []
    try:
        actor.save_weights(str(actor_temporary))
        critic.save_weights(str(critic_temporary))
        os.replace(actor_temporary, actor_path)
        committed.append(actor_path)
        os.replace(critic_temporary, critic_path)
        committed.append(critic_path)
    except Exception:
        for committed_path in committed:
            committed_path.unlink(missing_ok=True)
        raise
    finally:
        actor_temporary.unlink(missing_ok=True)
        critic_temporary.unlink(missing_ok=True)


def summarize_episodes(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = (
        "episode_return",
        "episode_length",
        "final_coverage",
        "coverage_auc",
        "collision_rate",
        "mean_policy_entropy",
    )
    summary: Dict[str, float] = {"n_episodes": float(len(records))}
    for key in keys:
        values = np.asarray([float(record.get(key, 0.0)) for record in records])
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"std_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    summary["success_rate"] = float(
        np.mean([bool(record.get("success", False)) for record in records])
    )
    for threshold_name in ("step_to_50", "step_to_75", "step_to_90"):
        reached = [
            int(record[threshold_name])
            for record in records
            if record.get(threshold_name) is not None
        ]
        suffix = threshold_name.removeprefix("step_to_")
        summary[f"reach_{suffix}_rate"] = float(len(reached) / len(records))
        summary[f"mean_{threshold_name}_when_reached"] = (
            float(np.mean(reached)) if reached else 0.0
        )
    return summary


def assert_finite_history(records: Sequence[Dict[str, object]]) -> None:
    numeric_keys = (
        "episode_return",
        "final_coverage",
        "mean_policy_entropy",
        "actor_loss",
        "critic_loss",
        "actor_grad_norm",
        "critic_grad_norm",
    )
    for episode_index, record in enumerate(records):
        for key in numeric_keys:
            if key in record and not np.isfinite(float(record[key])):
                raise FloatingPointError(
                    f"non-finite {key} at episode {episode_index}: {record[key]}"
                )


def package_versions() -> Dict[str, Optional[str]]:
    import gymnasium

    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "tensorflow": getattr(tf, "__version__", None),
        "numpy": np.__version__,
        "gymnasium": getattr(gymnasium, "__version__", None),
        "tensorflow_quantum": None,
        "cirq": None,
    }
    if quantum_dependencies_available():
        import cirq
        import tensorflow_quantum as tfq

        versions["tensorflow_quantum"] = getattr(tfq, "__version__", None)
        versions["cirq"] = getattr(cirq, "__version__", None)
    return versions


def build_reward_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        information_gain_scale=args.information_gain_scale,
        step_penalty=args.step_penalty,
        collision_penalty=args.collision_penalty,
        redundancy_penalty=args.redundancy_penalty,
        completion_bonus=args.completion_bonus,
    )


def create_environment(args: argparse.Namespace):
    return make_multi_agent_env_from_id(
        args.env_id,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        obstacle_density=args.obstacle_density,
        fov_range=args.fov_range,
        fov_angle=np.deg2rad(args.fov_angle_degrees),
        target_coverage=args.target_coverage,
        reward_config=build_reward_config(args),
    )


def create_models(
    framework: str,
    args: argparse.Namespace,
    obs_shape,
    actor_initial_weights: Optional[Sequence[np.ndarray]] = None,
):
    actor = generate_actor(obs_shape=obs_shape)
    if actor_initial_weights is not None:
        actor.set_weights([np.asarray(weight).copy() for weight in actor_initial_weights])
    critic_builder = get_critic_builder(framework)
    critic_kwargs = {
        "n_agents": args.n_agents,
        "obs_shape": obs_shape,
    }
    if framework_requires_quantum(framework):
        critic_kwargs.update(
            {
                "d_qubits": args.d_qubits,
                "n_layers": args.quantum_layers,
            }
        )
    critic = critic_builder(**critic_kwargs)
    return actor, critic



def model_weight_sha256(model: keras.Model) -> str:
    """Stable checksum for verifying paired model initialization."""
    digest = hashlib.sha256()
    for weight in model.get_weights():
        array = np.ascontiguousarray(weight)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()

def relative_weight_path(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def canonical_actor_initialization(
    model_seed: int,
    obs_shape: Sequence[int],
) -> tuple[List[np.ndarray], str]:
    """Build one canonical actor initialization for paired framework runs."""
    keras.backend.clear_session()
    seed_everything(model_seed)
    actor = generate_actor(obs_shape=tuple(int(value) for value in obs_shape))
    weights = [np.asarray(weight).copy() for weight in actor.get_weights()]
    checksum = model_weight_sha256(actor)
    del actor
    keras.backend.clear_session()
    gc.collect()
    return weights, checksum


def train_one_run(
    framework: str,
    model_seed: int,
    train_episode_seeds: Sequence[int],
    eval_episode_seeds: Sequence[int],
    run_id: str,
    args: argparse.Namespace,
    actor_initial_weights: Sequence[np.ndarray],
    expected_actor_initial_sha256: str,
) -> Dict[str, object]:
    keras.backend.clear_session()
    seed_everything(model_seed)
    environment = create_environment(args)
    _, size = parse_env_id(args.env_id)
    obs_shape = (size, size, OBS_CHANNELS)

    try:
        actor, critic = create_models(
            framework,
            args,
            obs_shape,
            actor_initial_weights=actor_initial_weights,
        )
        actor_initial_sha256 = model_weight_sha256(actor)
        if actor_initial_sha256 != expected_actor_initial_sha256:
            raise RuntimeError(
                "canonical actor initialization checksum mismatch before training"
            )
        critic_initial_sha256 = model_weight_sha256(critic)
        actor_optimizer, critic_optimizer = get_optimizer_configs(
            framework,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            quantum_critic_lr=args.quantum_critic_lr,
        )
        algorithm = MAA2C(
            env=environment,
            model_actor=actor,
            model_critic=critic,
            optimizer_actor=actor_optimizer,
            optimizer_critic=critic_optimizer,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            seed=model_seed,
        )
        train_action_seeds = make_seed_schedule(
            args.train_action_seed,
            model_seed=model_seed,
            count=len(train_episode_seeds),
        )
        train_history = algorithm.train(
            episode_seeds=train_episode_seeds,
            max_steps_per_episode=args.max_steps,
            progress_description=f"{framework} seed={model_seed}",
            show_progress=not args.no_progress,
            action_seeds=train_action_seeds,
        )
        assert_finite_history(train_history)

        # Evaluation maps and per-episode action RNG seeds are held fixed
        # across frameworks.  Per-episode seeding remains paired even when
        # policies produce different episode lengths.
        eval_action_seeds = make_seed_schedule(
            args.eval_action_seed,
            model_seed=model_seed,
            count=len(eval_episode_seeds),
        )
        eval_history = algorithm.evaluate(
            episode_seeds=eval_episode_seeds,
            deterministic=not args.stochastic_eval,
            max_steps_per_episode=args.max_steps,
            show_progress=False,
            action_seeds=eval_action_seeds,
        )
        assert_finite_history(eval_history)

        safe_env = args.env_id.replace("/", "-")
        actor_dir = args.output_dir / "actors" / run_id
        critic_dir = args.output_dir / "critics" / run_id
        actor_path = actor_dir / (
            f"actor-{framework}-{safe_env}-seed{model_seed}.weights.h5"
        )
        critic_path = critic_dir / (
            f"critic-{framework}-{safe_env}-seed{model_seed}.weights.h5"
        )
        save_weight_pair_atomically(actor, critic, actor_path, critic_path)
        actor_final_sha256 = model_weight_sha256(actor)
        critic_final_sha256 = model_weight_sha256(critic)

        return {
            "status": "completed",
            "run_id": run_id,
            "model_seed": int(model_seed),
            "initialization": {
                "actor_weights_sha256": actor_initial_sha256,
                "critic_weights_sha256": critic_initial_sha256,
            },
            "finalization": {
                "actor_weights_sha256": actor_final_sha256,
                "critic_weights_sha256": critic_final_sha256,
            },
            "parameters": {
                "actor": count_trainable_params(actor),
                "critic": count_trainable_params(critic),
                "total": count_trainable_params(actor) + count_trainable_params(critic),
            },
            "weights": {
                "actor": relative_weight_path(actor_path, args.output_dir),
                "critic": relative_weight_path(critic_path, args.output_dir),
                "actor_file_sha256": file_sha256(actor_path),
                "critic_file_sha256": file_sha256(critic_path),
            },
            "train_episode_seeds": [int(seed) for seed in train_episode_seeds],
            "train_action_seeds": [int(seed) for seed in train_action_seeds],
            "eval_episode_seeds": [int(seed) for seed in eval_episode_seeds],
            "eval_action_seeds": [int(seed) for seed in eval_action_seeds],
            "train": {
                "episodes": train_history,
                "summary": summarize_episodes(train_history),
                "last_100_summary": summarize_episodes(train_history[-100:]),
            },
            "evaluation": {
                "deterministic": not args.stochastic_eval,
                "episodes": eval_history,
                "summary": summarize_episodes(eval_history),
            },
        }
    finally:
        environment.close()
        keras.backend.clear_session()
        gc.collect()


def aggregate_framework_runs(
    framework_runs: Dict[str, Dict[str, object]]
) -> Dict[str, object]:
    completed = [
        run for run in framework_runs.values() if run.get("status") == "completed"
    ]
    if not completed:
        return {"n_completed_seeds": 0}

    metrics = (
        "mean_episode_return",
        "mean_final_coverage",
        "mean_coverage_auc",
        "success_rate",
        "mean_episode_length",
        "mean_collision_rate",
        "reach_50_rate",
        "reach_75_rate",
        "reach_90_rate",
    )
    aggregate: Dict[str, object] = {"n_completed_seeds": len(completed)}
    for metric in metrics:
        values = np.asarray(
            [
                float(run["evaluation"]["summary"].get(metric, 0.0))
                for run in completed
            ],
            dtype=np.float64,
        )
        aggregate[metric] = {
            "mean_across_seeds": float(np.mean(values)),
            "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "bootstrap_95_ci": bootstrap_mean_ci(values),
            "seed_values": values.tolist(),
        }
    parameter_values = [int(run["parameters"]["total"]) for run in completed]
    aggregate["total_parameters"] = parameter_values[0]
    return aggregate


def bootstrap_mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
) -> List[float]:
    """Deterministic seed-level percentile bootstrap interval for a mean."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return []
    if values.size == 1:
        value = float(values[0])
        return [value, value]
    rng = np.random.default_rng(20_260_722)
    indices = rng.integers(
        0,
        values.size,
        size=(int(n_resamples), values.size),
    )
    sample_means = values[indices].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(sample_means, [tail, 1.0 - tail])
    return [float(lower), float(upper)]


def config_for_json(args: argparse.Namespace) -> Dict[str, object]:
    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    return config


def initialize_results(
    args: argparse.Namespace,
    eval_episode_seeds: Sequence[int],
    run_id: str,
) -> Dict[str, object]:
    maze_type, size = parse_env_id(args.env_id)
    return {
        "schema_version": 3,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": package_versions(),
        "config": config_for_json(args),
        "environment": {
            "maze_type": maze_type,
            "size": size,
            "coordinate_convention": "arrays[y, x]",
            "observation_image_shape": [args.n_agents, size, size, OBS_CHANNELS],
            "observation_channels": list(OBSERVATION_CHANNELS),
            "aux_shape": [args.n_agents, AUX_DIM],
            "aux_semantics": ["x_normalized", "y_normalized", "east", "south", "west", "north"],
            "reward": build_reward_config(args).to_dict(),
        },
        "seed_protocol": {
            "training": (
                "unique child streams from SeedSequence([train_map_seed, model_seed]) "
                "for maps and SeedSequence([train_action_seed, model_seed]) for one action "
                "RNG per episode; both schedules are identical across frameworks "
                "at each model seed; shared actor initialization is verified by "
                "a pre-training SHA-256 checksum"
            ),
            "evaluation": (
                "SeedSequence([eval_map_seed]) with explicit exclusion of all "
                "training episode seeds; identical held-out maps for every "
                "framework and model seed"
            ),
            "stochastic_evaluation_actions": (
                "SeedSequence([eval_action_seed, model_seed]); one independent "
                "action RNG seed per evaluation episode, paired across frameworks"
            ),
            "eval_episode_seeds": [int(seed) for seed in eval_episode_seeds],
        },
        "framework_definitions": {
            framework: FRAMEWORK_DESCRIPTIONS[framework]
            for framework in args.frameworks
        },
        "runs": {framework: {} for framework in args.frameworks},
        "aggregate": {},
    }


def print_configuration(args: argparse.Namespace) -> None:
    _, size = parse_env_id(args.env_id)
    print(f"Environment: {args.env_id}", flush=True)
    print(f"Grid: {size}x{size}; agents: {args.n_agents}", flush=True)
    print(
        f"Training: {args.episodes} episodes x {args.n_seeds} model seeds; "
        f"max {args.max_steps} steps",
        flush=True,
    )
    print(
        f"Evaluation: {args.eval_episodes} held-out maps; "
        f"{'stochastic' if args.stochastic_eval else 'deterministic'} policy",
        flush=True,
    )
    print(f"Frameworks: {', '.join(args.frameworks)}", flush=True)


def run_benchmark(args: argparse.Namespace) -> Path:
    configure_tensorflow()
    # A temporary seed is required so that Conv2D/kernel random initialisers in
    # the parameter-count preview (before the real seed_everything call) do not
    # crash under tf.config.experimental.enable_op_determinism().
    tf.random.set_seed(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_training_seeds = set()
    for model_seed in range(
        args.model_seed_start,
        args.model_seed_start + args.n_seeds,
    ):
        all_training_seeds.update(
            make_seed_schedule(
                args.train_map_seed,
                model_seed=model_seed,
                count=args.episodes,
            )
        )
    eval_episode_seeds = make_disjoint_seed_schedule(
        args.eval_map_seed,
        count=args.eval_episodes,
        forbidden=all_training_seeds,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    results = initialize_results(args, eval_episode_seeds, run_id=run_id)
    safe_env = args.env_id.replace("/", "-")
    result_path = args.output_dir / f"slam-benchmark-{safe_env}-{run_id}.json"
    latest_path = args.output_dir / "latest.json"

    quantum_available = quantum_dependencies_available()
    if args.strict_quantum and any(
        framework_requires_quantum(framework) for framework in args.frameworks
    ) and not quantum_available:
        raise ImportError(
            "quantum frameworks were requested but TensorFlow Quantum is unavailable"
        )

    print_configuration(args)
    _, map_size = parse_env_id(args.env_id)
    obs_shape = (map_size, map_size, OBS_CHANNELS)
    canonical_actors: Dict[int, tuple[List[np.ndarray], str]] = {}

    # Print parameter count comparison for all requested frameworks
    print("\nParameter count comparison:", flush=True)
    print(f"  {'Framework':10s}  {'Actor':>10s}  {'Critic':>10s}  {'Total':>10s}", flush=True)
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}", flush=True)
    for framework in args.frameworks:
        if framework_requires_quantum(framework) and not quantum_available:
            print(f"  {framework:10s}  {'(skipped)':>10s}", flush=True)
            continue
        actor_tmp, critic_tmp = create_models(framework, args, obs_shape)
        actor_params = count_trainable_params(actor_tmp)
        critic_params = count_trainable_params(critic_tmp)
        print(
            f"  {framework:10s}  {actor_params:>10,d}  {critic_params:>10,d}  "
            f"{actor_params + critic_params:>10,d}",
            flush=True,
        )
        del actor_tmp, critic_tmp
        keras.backend.clear_session()
        gc.collect()
    print(flush=True)

    for model_seed in range(
        args.model_seed_start,
        args.model_seed_start + args.n_seeds,
    ):
        canonical_actors[model_seed] = canonical_actor_initialization(
            model_seed=model_seed,
            obs_shape=obs_shape,
        )

    for framework in args.frameworks:
        if framework_requires_quantum(framework) and not quantum_available:
            print(
                f"Skipping {framework}: TensorFlow Quantum dependencies are unavailable.",
                flush=True,
            )
            for model_seed in range(
                args.model_seed_start,
                args.model_seed_start + args.n_seeds,
            ):
                results["runs"][framework][str(model_seed)] = {
                    "status": "skipped_missing_quantum_dependency",
                    "model_seed": int(model_seed),
                }
            results["aggregate"][framework] = aggregate_framework_runs(
                results["runs"][framework]
            )
            atomic_write_json(result_path, results)
            atomic_write_json(latest_path, results)
            continue

        for model_seed in range(
            args.model_seed_start,
            args.model_seed_start + args.n_seeds,
        ):
            train_episode_seeds = make_seed_schedule(
                args.train_map_seed,
                model_seed=model_seed,
                count=args.episodes,
            )
            print("\n" + "=" * 72, flush=True)
            print(f"Training {framework} | model seed {model_seed}", flush=True)
            print("=" * 72, flush=True)
            try:
                run = train_one_run(
                    framework=framework,
                    model_seed=model_seed,
                    train_episode_seeds=train_episode_seeds,
                    eval_episode_seeds=eval_episode_seeds,
                    run_id=run_id,
                    args=args,
                    actor_initial_weights=canonical_actors[model_seed][0],
                    expected_actor_initial_sha256=canonical_actors[model_seed][1],
                )
            except Exception as exc:
                print(
                    f"ERROR: {framework} seed {model_seed} failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
                run = {
                    "status": "failed",
                    "model_seed": int(model_seed),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "train_episode_seeds": train_episode_seeds,
                }
                if args.fail_fast:
                    results["runs"][framework][str(model_seed)] = run
                    atomic_write_json(result_path, results)
                    atomic_write_json(latest_path, results)
                    raise

            results["runs"][framework][str(model_seed)] = run
            results["aggregate"][framework] = aggregate_framework_runs(
                results["runs"][framework]
            )
            atomic_write_json(result_path, results)
            atomic_write_json(latest_path, results)

    for framework in args.frameworks:
        results["aggregate"][framework] = aggregate_framework_runs(
            results["runs"][framework]
        )
    results["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(result_path, results)
    atomic_write_json(latest_path, results)

    print("\nBenchmark complete", flush=True)
    print(f"Results: {result_path}", flush=True)
    for framework in args.frameworks:
        aggregate = results["aggregate"].get(framework, {})
        completed = int(aggregate.get("n_completed_seeds", 0))
        if completed == 0:
            statuses = sorted(
                {
                    run.get("status", "unknown")
                    for run in results["runs"][framework].values()
                }
            )
            print(f"{framework:8s} | no completed runs ({', '.join(statuses)})")
            continue
        return_stats = aggregate["mean_episode_return"]
        coverage_stats = aggregate["mean_final_coverage"]
        success_stats = aggregate["success_rate"]
        print(
            f"{framework:8s} | eval return "
            f"{return_stats['mean_across_seeds']:.4f} ± "
            f"{return_stats['std_across_seeds']:.4f}; coverage "
            f"{coverage_stats['mean_across_seeds']:.3f}; success "
            f"{success_stats['mean_across_seeds']:.3f}",
            flush=True,
        )
    return result_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        args = normalize_args(parser.parse_args(argv))
        run_benchmark(args)
    except (ValueError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
