# Complete correction log

This file records the material changes made to the uploaded predecessor. It is also the modification notice for the CC BY 4.0 attribution requirement.

## 1. Dynamic and local observations

**Before:** the network consumed `self.grid.encode()`, a static full ground-truth map. Actor-side pose was only partially added in a later revision, while critic-side pose was discarded.

**Now:** `SLAM/slam_core.py` maintains one local belief map per agent. A cell remains unknown until it enters that agent's own FOV. The actor receives one local image plus its own pose/heading. Every critic receives the full collection of local images and auxiliary states during centralized training.

## 2. Semantic image encoding

**Before:** MiniGrid object/color/state category IDs were treated as RGB values and divided by 255. The explored channel therefore had a radically different scale.

**Now:** the observation is five explicit binary channels: unknown, free, wall, current FOV, and visited-by-self. The unknown/free/wall channels form a categorical one-hot partition.

## 3. Coordinate consistency

**Before:** the encoded grid was transposed while `explored_map` was not, silently pairing `[y, x]` cells with `[x, y]` exploration flags.

**Now:** every NumPy array uses `[y, x]`; Cartesian positions remain `(x, y)` only at API boundaries. No observation concatenation requires a transpose.

## 4. Critic auxiliary state

**Before:** MAA2C explicitly called the critic with images only.

**Now:** all critics have two inputs, `joint_image` and `joint_aux`. Position and heading are included in value estimation.

## 5. Synchronous multi-agent transitions

**Before:** agent 0 moved, changed the shared map, received reward, and generated an observation before agent 1 acted. Agent numbering changed both reward attribution and observation freshness.

**Now:** all movement proposals are computed from the same state, conflicts are resolved symmetrically, all FOVs are computed, all belief maps are updated once, one team reward is calculated, and every observation is generated from the same final joint state.

## 6. Distinct CTDE baselines

**Before:** `sCTDE` and `fCTDE` were the same `Flatten -> Dense -> value` network with different names.

**Now:** `sCTDE` has independent per-agent convolutional/dense branches before aggregation. `fCTDE` flattens the raw joint state into one centralized MLP. Architecture names, graph structures, and parameter counts differ.

## 7. Entropy regularization

**Before:** minimizing `policy_loss + alpha * entropy` suppressed entropy, and the benchmark used `alpha=0.05`.

**Now:** the update minimizes `policy_loss - entropy_coef * entropy`, with a default coefficient of `0.001`.

## 8. Stable actor-critic targets

**Before:** the critic target retained a gradient path through `V(next_state)`, Huber loss used a sum reduction, action probabilities were not clipped, and gradients were unbounded.

**Now:** values for GAE are computed outside gradient tapes, targets and advantages are detached, losses are means, probabilities are clipped and renormalized, advantages can be normalized, and actor/critic gradients use global-norm clipping. True terminals stop bootstrapping; time-limit truncations bootstrap.

## 9. Turning and collision semantics

**Before:** any unchanged position—including a valid left or right turn—received a large standing-still penalty.

**Now:** movement outcomes separately record `turned`, `moved`, `forward_attempt`, and `collision`. Valid turning incurs only the ordinary time cost. Collision penalties apply only to blocked forward movement or multi-agent movement conflicts.

## 10. Reward scale

**Before:** convex information gain and area-scaled milestone bonuses produced very large, map-size-dependent targets.

**Now:** information gain is linear and normalized by the number of observable target cells. Milestone and completion bonuses are bounded constants. Default cumulative reward components remain comparable across map sizes.

## 11. Reachability and termination

**Before:** random obstacles could create unreachable regions, while success required exploring every cell in `size * size`.

**Now:** random obstacles are accepted only when free space remains connected. The target mask comprises reachable free cells and observable adjacent walls. Success uses configurable coverage of that mask, defaulting to 95%.

## 12. Reproducibility and fair comparison

**Before:** NumPy and TensorFlow seeds did not seed Gym map generation; each framework could see a different map stream; one seed was treated as a benchmark.

**Now:** every reset accepts an explicit map seed. Frameworks share paired training-map schedules and paired per-episode action RNG schedules for each independent model seed. One canonical actor initialization is copied into every framework and verified before training by SHA-256. All runs share a held-out evaluation schedule whose seed values are excluded from training schedules. Aggregates operate across model seeds and include deterministic bootstrap confidence intervals.

## 13. Environment ID handling

**Before:** the training factory parsed map size but hard-coded `maze_type="random"`.

**Now:** `EmptySLAM`, `HardSLAM`, and `RandomSLAM` IDs select the corresponding generator.

## 14. Missing submodule

**Before:** the archive contained an empty `eqmarl/` directory and an unusable `.gitmodules` reference, so installation was incomplete.

**Now:** the required MAA2C algorithm and quantum layers are included directly under `SLAM/`. The submodule declaration and empty directory are removed. `pyproject.toml` provides an installable package and CLI entry point.

## 15. Incompatible artifacts and notebook

**Before:** bundled results, actor weights, figures, README claims, and notebook cells belonged to an older three-channel actor and could not validate the revised code.

**Now:** all stale generated artifacts are removed. Results directories contain placeholders only. The notebook reads schema version 3 benchmark JSON and has no embedded outputs. The README makes no empirical ranking claim.

## 16. Quantum bottleneck and baseline semantics

**Before:** the revised quantum critic used a large generic dense preprocessor, while the older version compressed thousands of inputs directly to a few angles and measured one global observable. `qfCTDE` semantics were not clearly distinct from eQMARL.

**Now:** eQMARL uses local spatial encoders and local angle partitions with optional initial \(\Psi^+\) cross-agent entanglement. qfCTDE uses a centralized preprocessor and a globally entangling circuit. Both use the same richer readout set: global parity, per-agent parity, and every single-qubit Z expectation. Parameter counts are recorded rather than described as inherently efficient.

## 17. Interface and operational failures

**Before:** a vector-like wrapper declared a `Box` but returned a tuple, README CLI flags were ignored, environment types were not honored, and a failed run could trigger an `IndexError` in final reporting.

**Now:** the joint Gymnasium environment has a correct `Dict` observation and scalar team reward. The compatibility adapter declares a matching `Tuple`. `argparse` handles all documented flags and environment variables. Run status is explicit, JSON is written atomically, skipped/failed runs remain reportable, and summary code never indexes an empty history.

## Additional observability

Episode records now include return, length, final coverage, fixed-horizon coverage AUC, 50/75/90% coverage times, success, collisions, turns, moves, policy entropy, actor/critic losses, gradient norms, advantage statistics, and parameter counts.

## 18. Input validation and numerical failure detection

**Before:** several interfaces silently cast floating actions to integers, accepted NaN hyperparameters, or allowed a non-finite loss to contaminate weights before failure became visible.

**Now:** environment actions and trajectory actions must use integer dtypes and lie in range. Seeds, reward coefficients, optimizer settings, observations, rewards, terminal masks, critic bootstraps, losses, gradients, and global gradient norms are validated. Non-finite values fail at the transition or update that produced them.

## 19. Artifact isolation

**Before:** repeated runs could overwrite identically named weight files, while result JSON might continue to reference ambiguous paths.

**Now:** every benchmark receives a microsecond-resolution run ID. Actor and critic weights are saved in run-specific directories, weight-file hashes are recorded in JSON, and stale predecessor artifacts are excluded from the release.


## 20. NumPy categorical-sampling normalization hotfix

**Before:** actor probabilities were normalized in TensorFlow float32 and then
cast to NumPy float64. A valid row such as three copies of `0.33333334` then
summed to approximately `1.00000003`, exceeding `numpy.random.Generator.choice`'s
stricter float64 tolerance. Every framework could therefore fail on its first
stochastic action with `ValueError: probabilities do not sum to 1`.

**Now:** inference probabilities are converted to float64 first, clipped, and
renormalized in NumPy immediately before categorical sampling. The remaining
round-off residual is absorbed into the largest component, and a regression test
covers the exact float32 one-third case observed on GPU.
