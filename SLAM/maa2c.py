"""Stable self-contained multi-agent advantage actor-critic implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from .numerics import _normalize_sampling_probabilities


@dataclass(frozen=True)
class UpdateStats:
    actor_loss: float
    critic_loss: float
    policy_loss: float
    entropy: float
    actor_grad_norm: float
    critic_grad_norm: float
    advantage_mean: float
    advantage_std: float
    value_mean: float

    def to_dict(self) -> Dict[str, float]:
        return {
            name: float(value)
            for name, value in self.__dict__.items()
        }


class MAA2C:
    """Shared decentralized actor with one centralized state-value critic.

    Updates are on-policy and trajectory based.  Generalized advantage
    estimation is used, targets are detached, losses are averaged, the entropy
    term has the exploration-promoting sign, and both optimizers use global
    norm clipping.
    """

    def __init__(
        self,
        env,
        model_actor: keras.Model,
        model_critic: keras.Model,
        optimizer_actor: keras.optimizers.Optimizer,
        optimizer_critic: keras.optimizers.Optimizer,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.001,
        max_grad_norm: float = 1.0,
        probability_epsilon: float = 1e-7,
        normalize_advantages: bool = True,
        seed: int = 0,
    ) -> None:
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not np.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be finite and in [0, 1]")
        if not np.isfinite(entropy_coef) or entropy_coef < 0.0:
            raise ValueError("entropy_coef must be finite and non-negative")
        if not np.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        if (
            not np.isfinite(probability_epsilon)
            or not 0.0 < probability_epsilon < 1.0
        ):
            raise ValueError("probability_epsilon must be finite and in (0, 1)")
        if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
            raise ValueError("seed must be a non-negative integer")

        self.env = env
        self.model_actor = model_actor
        self.model_critic = model_critic
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.probability_epsilon = float(probability_epsilon)
        self.normalize_advantages = bool(normalize_advantages)
        self.rng = np.random.default_rng(int(seed))

        self.n_agents = int(getattr(env, "n_agents"))
        nvec = np.asarray(env.action_space.nvec, dtype=np.int64)
        if nvec.shape != (self.n_agents,) or not np.all(nvec == nvec[0]):
            raise ValueError(
                "MAA2C requires one equal-size discrete action space per agent"
            )
        self.n_actions = int(nvec[0])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _validate_observation(
        self, observation: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (
            not isinstance(observation, dict)
            or "image" not in observation
            or "aux" not in observation
        ):
            raise TypeError("observation must contain 'image' and 'aux'")
        images = np.asarray(observation["image"], dtype=np.float32)
        aux = np.asarray(observation["aux"], dtype=np.float32)
        if images.ndim != 4 or images.shape[0] != self.n_agents:
            raise ValueError(
                "observation image must have shape "
                f"(n_agents, height, width, channels); got {images.shape}"
            )
        if aux.ndim != 2 or aux.shape[0] != self.n_agents:
            raise ValueError(
                "observation aux must have shape (n_agents, features); "
                f"got {aux.shape}"
            )
        if not np.all(np.isfinite(images)) or not np.all(np.isfinite(aux)):
            raise FloatingPointError("observation contains non-finite values")
        return images, aux

    @staticmethod
    def _copy_observation(
        observation: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return {
            "image": np.asarray(observation["image"], dtype=np.float32).copy(),
            "aux": np.asarray(observation["aux"], dtype=np.float32).copy(),
        }

    def action_probabilities(
        self, observation: Dict[str, np.ndarray]
    ) -> np.ndarray:
        image_array, aux_array = self._validate_observation(observation)
        images = tf.convert_to_tensor(image_array, dtype=tf.float32)
        aux = tf.convert_to_tensor(aux_array, dtype=tf.float32)
        probabilities = self.model_actor([images, aux], training=False)
        tf.debugging.check_numerics(
            probabilities, "actor produced non-finite action probabilities"
        )
        if probabilities.shape.rank != 2:
            raise ValueError(
                f"actor output must be rank 2, got rank {probabilities.shape.rank}"
            )
        if probabilities.shape[0] not in (None, self.n_agents):
            raise ValueError(
                "actor output batch dimension does not match the number of agents"
            )
        if int(tf.shape(probabilities)[0].numpy()) != self.n_agents:
            raise ValueError(
                "actor returned the wrong number of per-agent distributions"
            )
        if probabilities.shape[-1] not in (None, self.n_actions):
            raise ValueError(
                "actor output action dimension does not match the environment"
            )
        return _normalize_sampling_probabilities(
            probabilities.numpy(), self.probability_epsilon
        )

    def policy(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        probabilities = self.action_probabilities(observation)
        if deterministic:
            actions = np.argmax(probabilities, axis=-1).astype(np.int64)
        else:
            action_rng = self.rng if rng is None else rng
            actions = np.asarray(
                [
                    action_rng.choice(self.n_actions, p=agent_probabilities)
                    for agent_probabilities in probabilities
                ],
                dtype=np.int64,
            )
        return actions, probabilities.astype(np.float32)

    def value(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        image_array, aux_array = self._validate_observation(observation)
        images = tf.convert_to_tensor(
            image_array[None, ...], dtype=tf.float32
        )
        aux = tf.convert_to_tensor(
            aux_array[None, ...], dtype=tf.float32
        )
        value = self.model_critic([images, aux], training=False)
        tf.debugging.check_numerics(value, "critic produced a non-finite value")
        flat_value = tf.reshape(value, (-1,))
        if int(tf.size(flat_value).numpy()) != self.n_agents:
            raise ValueError("critic must return one value per agent")
        return flat_value.numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _stack_observations(
        self, observations: Sequence[Dict[str, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        validated = [self._validate_observation(observation) for observation in observations]
        return (
            np.stack([images for images, _ in validated]).astype(np.float32),
            np.stack([aux for _, aux in validated]).astype(np.float32),
        )

    def _gae_targets(
        self,
        rewards: np.ndarray,
        terminal_flags: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T, n_agents = rewards.shape
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        for a in range(n_agents):
            accumulator = 0.0
            for index in range(T - 1, -1, -1):
                continuation = 1.0 - float(terminal_flags[index])
                delta = (
                    rewards[index, a]
                    + self.gamma * continuation * next_values[index, a]
                    - values[index, a]
                )
                accumulator = delta + (
                    self.gamma * self.gae_lambda * continuation * accumulator
                )
                advantages[index, a] = accumulator
            returns[:, a] = advantages[:, a] + values[:, a]
        return advantages.astype(np.float32), returns.astype(np.float32)

    @staticmethod
    def _clip_and_apply(
        optimizer: keras.optimizers.Optimizer,
        gradients,
        variables,
        max_grad_norm: float,
    ) -> float:
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        if not pairs:
            raise RuntimeError("model update produced no gradients")
        gradient_list, variable_list = zip(*pairs)
        checked_gradients = [
            tf.debugging.check_numerics(
                gradient,
                f"non-finite gradient for {variable.name}",
            )
            for gradient, variable in zip(gradient_list, variable_list)
        ]
        clipped_gradients, global_norm = tf.clip_by_global_norm(
            checked_gradients, clip_norm=max_grad_norm
        )
        tf.debugging.check_numerics(global_norm, "non-finite global gradient norm")
        optimizer.apply_gradients(zip(clipped_gradients, variable_list))
        return float(global_norm.numpy())

    def update(
        self,
        states: Sequence[Dict[str, np.ndarray]],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: Sequence[Dict[str, np.ndarray]],
        terminal_flags: np.ndarray,
    ) -> UpdateStats:
        if not states:
            raise ValueError("cannot update from an empty trajectory")

        state_images_np, state_aux_np = self._stack_observations(states)
        next_images_np, next_aux_np = self._stack_observations(next_states)
        raw_actions = np.asarray(actions)
        if not np.issubdtype(raw_actions.dtype, np.integer):
            raise TypeError("trajectory actions must use an integer dtype")
        actions_np = raw_actions.astype(np.int32, copy=False)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        if rewards_np.ndim == 1:
            rewards_np = rewards_np[:, None]
        terminal_np = np.asarray(terminal_flags, dtype=np.float32).reshape(-1)
        trajectory_length = len(states)
        if len(next_states) != trajectory_length:
            raise ValueError("states and next_states must have equal length")
        if actions_np.shape != (trajectory_length, self.n_agents):
            raise ValueError(
                "actions must have shape "
                f"({trajectory_length}, {self.n_agents}), got {actions_np.shape}"
            )
        if rewards_np.shape != (trajectory_length, self.n_agents):
            raise ValueError(
                f"per-agent rewards must have shape "
                f"({trajectory_length}, {self.n_agents}), got {rewards_np.shape}"
            )
        if terminal_np.shape != (trajectory_length,):
            raise ValueError("one terminal flag is required per transition")
        if np.any((actions_np < 0) | (actions_np >= self.n_actions)):
            raise ValueError(
                f"trajectory actions must be in [0, {self.n_actions - 1}]"
            )
        for name, array in (
            ("state images", state_images_np),
            ("state auxiliary values", state_aux_np),
            ("next-state images", next_images_np),
            ("next-state auxiliary values", next_aux_np),
            ("rewards", rewards_np),
            ("terminal flags", terminal_np),
        ):
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"{name} contain non-finite values")
        if np.any((terminal_np != 0.0) & (terminal_np != 1.0)):
            raise ValueError("terminal flags must contain only 0 or 1")

        state_images = tf.convert_to_tensor(state_images_np, dtype=tf.float32)
        state_aux = tf.convert_to_tensor(state_aux_np, dtype=tf.float32)
        next_images = tf.convert_to_tensor(next_images_np, dtype=tf.float32)
        next_aux = tf.convert_to_tensor(next_aux_np, dtype=tf.float32)

        # Target computation is outside gradient tapes; both target and
        # advantage are constants during optimization.
        values = tf.reshape(
            self.model_critic([state_images, state_aux], training=False),
            (trajectory_length, self.n_agents),
        ).numpy().astype(np.float32)
        next_values = tf.reshape(
            self.model_critic([next_images, next_aux], training=False),
            (trajectory_length, self.n_agents),
        ).numpy().astype(np.float32)
        if values.shape != (trajectory_length, self.n_agents) or next_values.shape != (
            trajectory_length, self.n_agents,
        ):
            raise ValueError(
                "critic must return one value per agent for every trajectory state"
            )
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(next_values)):
            raise FloatingPointError("critic produced non-finite bootstrap values")
        raw_advantages, returns = self._gae_targets(
            rewards_np, terminal_np, values, next_values
        )
        if not np.all(np.isfinite(raw_advantages)) or not np.all(np.isfinite(returns)):
            raise FloatingPointError("GAE produced non-finite targets")
        advantages = raw_advantages.copy()
        if self.normalize_advantages and advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        advantages_tensor = tf.stop_gradient(
            tf.convert_to_tensor(advantages, dtype=tf.float32)
        )
        returns_tensor = tf.stop_gradient(
            tf.convert_to_tensor(returns, dtype=tf.float32)
        )
        actions_tensor = tf.convert_to_tensor(actions_np, dtype=tf.int32)

        image_shape = state_images.shape.as_list()[2:]
        aux_dim = int(state_aux.shape.as_list()[-1])
        actor_images = tf.reshape(state_images, [-1] + image_shape)
        actor_aux = tf.reshape(state_aux, [-1, aux_dim])
        flat_actions = tf.reshape(actions_tensor, (-1,))
        per_agent_advantages = tf.reshape(advantages_tensor, (-1,))

        with tf.GradientTape() as actor_tape:
            probabilities = self.model_actor(
                [actor_images, actor_aux], training=True
            )
            probabilities = tf.clip_by_value(
                probabilities, self.probability_epsilon, 1.0
            )
            probabilities = probabilities / tf.reduce_sum(
                probabilities, axis=-1, keepdims=True
            )
            row_indices = tf.range(
                tf.shape(flat_actions)[0], dtype=tf.int32
            )
            chosen_indices = tf.stack(
                [row_indices, flat_actions], axis=1
            )
            chosen_probabilities = tf.gather_nd(
                probabilities, chosen_indices
            )
            log_probabilities = tf.math.log(chosen_probabilities)
            policy_loss = -tf.reduce_mean(
                log_probabilities * per_agent_advantages
            )
            entropy = -tf.reduce_mean(
                tf.reduce_sum(
                    probabilities * tf.math.log(probabilities), axis=-1
                )
            )
            # The negative sign is required when minimizing the loss.
            actor_loss = policy_loss - self.entropy_coef * entropy
            tf.debugging.check_numerics(actor_loss, "non-finite actor loss")

        actor_gradients = actor_tape.gradient(
            actor_loss, self.model_actor.trainable_variables
        )
        actor_grad_norm = self._clip_and_apply(
            self.optimizer_actor,
            actor_gradients,
            self.model_actor.trainable_variables,
            self.max_grad_norm,
        )

        huber = keras.losses.Huber(reduction=keras.losses.Reduction.NONE)
        with tf.GradientTape() as critic_tape:
            predicted_values = self.model_critic(
                [state_images, state_aux], training=True
            )
            predicted_values = tf.reshape(
                predicted_values, (trajectory_length, self.n_agents)
            )
            critic_loss = tf.reduce_mean(
                huber(returns_tensor, predicted_values)
            )
            tf.debugging.check_numerics(critic_loss, "non-finite critic loss")

        critic_gradients = critic_tape.gradient(
            critic_loss, self.model_critic.trainable_variables
        )
        critic_grad_norm = self._clip_and_apply(
            self.optimizer_critic,
            critic_gradients,
            self.model_critic.trainable_variables,
            self.max_grad_norm,
        )

        return UpdateStats(
            actor_loss=float(actor_loss.numpy()),
            critic_loss=float(critic_loss.numpy()),
            policy_loss=float(policy_loss.numpy()),
            entropy=float(entropy.numpy()),
            actor_grad_norm=actor_grad_norm,
            critic_grad_norm=critic_grad_norm,
            advantage_mean=float(np.mean(raw_advantages)),
            advantage_std=float(np.std(raw_advantages)),
            value_mean=float(np.mean(values)),
        )

    # ------------------------------------------------------------------
    # Episode loops
    # ------------------------------------------------------------------
    @staticmethod
    def _episode_record_from_metrics(
        seed: Optional[int],
        metrics: Dict[str, object],
        episode_return: float,
        episode_length: int,
        mean_entropy: float,
        terminated: bool,
        truncated: bool,
        coverage_auc: float,
        step_to_50: Optional[int],
        step_to_75: Optional[int],
        step_to_90: Optional[int],
    ) -> Dict[str, object]:
        return {
            "seed": None if seed is None else int(seed),
            "episode_return": float(episode_return),
            "episode_length": int(episode_length),
            "mean_policy_entropy": float(mean_entropy),
            "final_coverage": float(metrics.get("coverage", 0.0)),
            "success": bool(metrics.get("success", terminated)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "coverage_auc": float(coverage_auc),
            "step_to_50": None if step_to_50 is None else int(step_to_50),
            "step_to_75": None if step_to_75 is None else int(step_to_75),
            "step_to_90": None if step_to_90 is None else int(step_to_90),
            "collision_rate": float(metrics.get("collision_rate", 0.0)),
            "collisions_total": int(metrics.get("collisions_total", 0)),
            "turns_total": int(metrics.get("turns_total", 0)),
            "moves_total": int(metrics.get("moves_total", 0)),
            "target_cells": int(metrics.get("target_cells", 0)),
            "explored_cells": int(metrics.get("explored_cells", 0)),
        }

    def run_episode(
        self,
        seed: Optional[int],
        training: bool,
        deterministic: bool = False,
        max_steps: Optional[int] = None,
        action_seed: Optional[int] = None,
    ) -> Dict[str, object]:
        observation, reset_info = self.env.reset(seed=seed)
        episode_rng = (
            np.random.default_rng(int(action_seed))
            if action_seed is not None
            else None
        )
        initial_metrics = dict(reset_info.get("metrics", {}))
        if bool(initial_metrics.get("success", False)):
            record = self._episode_record_from_metrics(
                seed=seed,
                metrics=initial_metrics,
                episode_return=0.0,
                episode_length=0,
                mean_entropy=0.0,
                terminated=True,
                truncated=False,
                coverage_auc=float(initial_metrics.get("coverage", 0.0)),
                step_to_50=(0 if float(initial_metrics.get("coverage", 0.0)) >= 0.50 else None),
                step_to_75=(0 if float(initial_metrics.get("coverage", 0.0)) >= 0.75 else None),
                step_to_90=(0 if float(initial_metrics.get("coverage", 0.0)) >= 0.90 else None),
            )
            record["action_seed"] = (
                None if action_seed is None else int(action_seed)
            )
            return record

        step_limit = int(self.env.max_steps if max_steps is None else max_steps)
        if step_limit < 1:
            raise ValueError("max_steps must be positive")
        states: List[Dict[str, np.ndarray]] = []
        next_states: List[Dict[str, np.ndarray]] = []
        action_history: List[np.ndarray] = []
        reward_history: List[np.ndarray] = []
        terminal_history: List[float] = []
        entropy_history: List[float] = []
        initial_coverage = float(initial_metrics.get("coverage", 0.0))
        coverage_history: List[float] = [initial_coverage]
        final_info: Dict[str, object] = dict(reset_info)
        terminated = False
        truncated = False

        for _ in range(step_limit):
            actions, probabilities = self.policy(
                observation, deterministic=deterministic, rng=episode_rng
            )
            next_observation, reward, terminated, truncated, info = self.env.step(
                actions
            )
            per_agent_reward = np.asarray(reward, dtype=np.float32)
            states.append(self._copy_observation(observation))
            next_states.append(self._copy_observation(next_observation))
            action_history.append(actions.copy())
            reward_history.append(per_agent_reward)
            # Bootstrap through a time-limit truncation, not a true terminal.
            terminal_history.append(float(terminated))
            entropy_history.append(
                float(
                    -np.mean(
                        np.sum(
                            probabilities
                            * np.log(np.clip(probabilities, 1e-7, 1.0)),
                            axis=-1,
                        )
                    )
                )
            )
            observation = next_observation
            final_info = dict(info)
            coverage_history.append(
                float(info.get("metrics", {}).get("coverage", coverage_history[-1]))
            )
            if terminated or truncated:
                break

        if not terminated and not truncated and len(reward_history) >= step_limit:
            truncated = True

        update_stats: Optional[UpdateStats] = None
        if training and states:
            update_stats = self.update(
                states=states,
                actions=np.stack(action_history),
                rewards=np.stack(reward_history),
                next_states=next_states,
                terminal_flags=np.asarray(terminal_history, dtype=np.float32),
            )

        metrics = dict(final_info.get("metrics", self.env.get_metrics()))
        # Compare exploration speed over a common fixed horizon.  The final
        # coverage is held constant after early termination, which rewards
        # reaching high coverage sooner without favoring longer episodes.
        if len(coverage_history) < step_limit + 1:
            coverage_history.extend(
                [coverage_history[-1]] * (step_limit + 1 - len(coverage_history))
            )
        coverage_curve = np.asarray(coverage_history[: step_limit + 1], dtype=np.float64)
        coverage_auc = float(np.trapz(coverage_curve, dx=1.0) / step_limit)

        def first_threshold_step(threshold: float) -> Optional[int]:
            reached = np.flatnonzero(coverage_curve >= threshold)
            return int(reached[0]) if reached.size else None

        record = self._episode_record_from_metrics(
            seed=seed,
            metrics=metrics,
            episode_return=float(np.sum(np.mean(np.stack(reward_history), axis=-1))) if reward_history else 0.0,
            episode_length=len(reward_history),
            mean_entropy=(
                float(np.mean(entropy_history)) if entropy_history else 0.0
            ),
            terminated=terminated,
            truncated=truncated,
            coverage_auc=coverage_auc,
            step_to_50=first_threshold_step(0.50),
            step_to_75=first_threshold_step(0.75),
            step_to_90=first_threshold_step(0.90),
        )
        if update_stats is not None:
            record.update(update_stats.to_dict())
        record["action_seed"] = None if action_seed is None else int(action_seed)
        return record

    def train(
        self,
        episode_seeds: Sequence[int],
        max_steps_per_episode: Optional[int] = None,
        progress_description: str = "training",
        show_progress: bool = True,
        action_seeds: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, object]]:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            tqdm = None

        seeds = [int(seed) for seed in episode_seeds]
        paired_action_seeds: List[Optional[int]] = (
            [None] * len(seeds)
            if action_seeds is None
            else [int(seed) for seed in action_seeds]
        )
        if len(paired_action_seeds) != len(seeds):
            raise ValueError("action_seeds and episode_seeds must have equal length")
        pairs = list(zip(seeds, paired_action_seeds))
        iterator = pairs
        if show_progress and tqdm is not None:
            iterator = tqdm(pairs, desc=progress_description)

        history: List[Dict[str, object]] = []
        for episode_index, (episode_seed, action_seed) in enumerate(iterator):
            record = self.run_episode(
                seed=episode_seed,
                training=True,
                deterministic=False,
                max_steps=max_steps_per_episode,
                action_seed=action_seed,
            )
            record["episode"] = int(episode_index)
            history.append(record)
            if (
                show_progress
                and tqdm is not None
                and hasattr(iterator, "set_postfix")
            ):
                iterator.set_postfix(
                    ret=f"{record['episode_return']:.3f}",
                    cov=f"{record['final_coverage']:.3f}",
                    ent=f"{record['mean_policy_entropy']:.3f}",
                )
        return history

    def evaluate(
        self,
        episode_seeds: Sequence[int],
        deterministic: bool = True,
        max_steps_per_episode: Optional[int] = None,
        show_progress: bool = False,
        progress_description: str = "evaluation",
        action_seeds: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, object]]:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            tqdm = None

        seeds = [int(seed) for seed in episode_seeds]
        paired_action_seeds: List[Optional[int]] = (
            [None] * len(seeds)
            if action_seeds is None
            else [int(seed) for seed in action_seeds]
        )
        if len(paired_action_seeds) != len(seeds):
            raise ValueError("action_seeds and episode_seeds must have equal length")
        pairs = list(zip(seeds, paired_action_seeds))
        iterator = pairs
        if show_progress and tqdm is not None:
            iterator = tqdm(pairs, desc=progress_description)

        history: List[Dict[str, object]] = []
        for episode_index, (episode_seed, action_seed) in enumerate(iterator):
            record = self.run_episode(
                seed=episode_seed,
                training=False,
                deterministic=deterministic,
                max_steps=max_steps_per_episode,
                action_seed=action_seed,
            )
            record["episode"] = int(episode_index)
            record["action_seed"] = (
                None if action_seed is None else int(action_seed)
            )
            history.append(record)
        return history


__all__ = ["MAA2C", "UpdateStats"]
