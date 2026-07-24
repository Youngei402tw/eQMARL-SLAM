# eQMARL-SLAM — corrected cooperative SLAM benchmark

This repository applies the eQMARL critic concept to a cooperative, partially observable mapping task and compares four centralized-training/decentralized-execution (CTDE) critics:

| Framework | Critic implementation |
|---|---|
| `eqmarl` | Independent per-agent classical encoders feed local quantum partitions initialized with cross-agent \(\Psi^+\) entanglement. |
| `qfctde` | The raw joint state is processed centrally and encoded into one fully centralized variational quantum circuit. |
| `sctde` | Independent classical per-agent branches are aggregated after local encoding. |
| `fctde` | One classical network processes the raw joint observation centrally. |

All frameworks use the same shared decentralized actor. This corrected release is self-contained; it no longer depends on the missing `eqmarl` Git submodule.

## What was corrected

The uploaded predecessor mixed static ground-truth maps with dynamic exploration state, discarded pose information in the critic, updated agents sequentially, treated turning as failure, used identical `sCTDE` and `fCTDE` models, and contained unstable A2C losses. It also bundled results and weights produced by an incompatible older observation interface.

The current implementation addresses those failures:

- no ground-truth map is exposed to the actor or critic;
- every agent receives its own local semantic belief map;
- all arrays use the single convention `array[y, x]`;
- image channels are binary semantics, not category IDs divided by 255;
- pose and heading enter both actor and critic;
- joint movement, FOV updates, mapping, reward, and observations are synchronous;
- normal turning is not a collision or standing-still penalty;
- maps are generated with connected free space and success is defined over observable reachable cells;
- reward magnitudes are normalized by map size;
- `sCTDE` and `fCTDE` are genuinely different architectures;
- the entropy term has the correct exploration-promoting sign;
- GAE targets are detached, losses are averaged, probabilities are clipped, and gradients use global-norm clipping;
- framework comparisons use paired training-map schedules and common held-out evaluation maps;
- uncertainty is aggregated across independent model seeds, including seed-level bootstrap intervals;
- stale JSON, weights, plots, and notebook outputs are removed.

A line-by-line issue mapping is in [`CHANGELOG_FIXES.md`](CHANGELOG_FIXES.md).

## Environment

`MultiAgentSLAMEnv` is one joint Gymnasium environment. Its action is a `MultiDiscrete([3] * n_agents)` vector and its reward is one scalar team reward.

Each decentralized agent observes:

```text
image: (height, width, 5)
  0 unknown
  1 known free
  2 known wall
  3 current field of view
  4 cells visited by this agent

aux: (6,)
  normalized x, normalized y, east, south, west, north
```

The first three channels are one-hot at every cell. An agent learns a cell only when it enters that agent's line of sight. The centralized critic receives the collection of all local observations during training; the actor receives only one agent's local observation.

The simulator assumes perfect agent localization: normalized position and heading are supplied explicitly. It therefore benchmarks cooperative mapping and exploration under known pose, not pose-estimation uncertainty or loop-closure optimization.

Map IDs are honored:

```text
MiniGrid-EmptySLAM-{6,8,16,32}x{6,8,16,32}-v0
MiniGrid-HardSLAM-{6,8,16,32}x{6,8,16,32}-v0
MiniGrid-RandomSLAM-{6,8,16,32}x{6,8,16,32}-v0
```

The historical `MiniGrid-...` prefix is retained for compatibility, but MiniGrid itself is no longer required.

## Installation

The reference eQMARL software stack pins TensorFlow 2.7.0 and TensorFlow Quantum 0.7.2, so use Python 3.9.

### Complete quantum environment

```bash
conda env create -f environment.yml
conda activate eqmarl_slam
python -m SLAM.smoke_test
```

Equivalent pip installation from an existing Python 3.9 environment:

```bash
python -m pip install -U pip
python -m pip install -e ".[quantum,analysis,test]"
```

### Classical-only environment

This runs `sctde` and `fctde` without Cirq or TensorFlow Quantum:

```bash
python -m pip install -e ".[analysis,test]"
python -m SLAM.smoke_test
```

## Tests

```bash
python -m pytest
```

The test suite is dependency-aware:

- pure NumPy environment tests always run;
- Gymnasium interface tests run when Gymnasium is installed;
- classical model and MAA2C tests run when TensorFlow is installed;
- quantum forward-pass tests run when Cirq and TensorFlow Quantum are installed.

## Training

### Fast classical smoke benchmark

```bash
python -m SLAM.run_benchmark \
  --quick \
  --env-id MiniGrid-RandomSLAM-6x6-v0 \
  --frameworks sctde fctde
```

### Fast all-framework smoke benchmark

```bash
python -m SLAM.run_benchmark \
  --quick \
  --env-id MiniGrid-RandomSLAM-6x6-v0 \
  --frameworks eqmarl qfctde sctde fctde \
  --strict-quantum
```

Without `--strict-quantum`, unavailable quantum frameworks are recorded as skipped while classical runs continue.

### Full comparison

```bash
python -m SLAM.run_benchmark \
  --env-id MiniGrid-RandomSLAM-32x32-v0 \
  --frameworks eqmarl qfctde sctde fctde \
  --n-seeds 10 \
  --episodes 1000 \
  --max-steps 200 \
  --eval-episodes 100 \
  --strict-quantum
```

The CLI accepts the former `--n-episodes` spelling as an alias for `--episodes`. Environment-variable equivalents remain available, including `SLAM_ENV_ID`, `SLAM_N_SEEDS`, `SLAM_N_EPISODES`, `SLAM_MAX_STEPS`, `SLAM_FRAMEWORKS`, and `SLAM_OUTPUT_DIR`.

## Experimental protocol

For every model seed:

1. each framework receives the same deterministic training-map seed sequence;
2. one canonical actor weight set is created per model seed, copied into every framework, and verified by SHA-256 before training;
3. each training episode also receives a paired action-sampling RNG seed, so stochastic action streams are controlled independently of episode length;
4. evaluation uses the same held-out map seeds for every framework and every model seed;
5. evaluation seeds are explicitly excluded from all configured training seed schedules;
6. stochastic evaluation, when enabled, uses one paired action RNG seed per episode; deterministic evaluation remains the default;
7. aggregate uncertainty is calculated over independent model seeds, not adjacent episodes from one training trajectory.

Primary evaluation metrics include return, final coverage, fixed-horizon coverage AUC, success rate, steps to 50/75/90% coverage, collision rate, policy entropy, episode length, parameter count, losses, and gradient norms.

## Outputs

By default, generated artifacts are written to `SLAM/results/`:

```text
latest.json
slam-benchmark-<environment>-<run-id>.json
actors/<run-id>/*.weights.h5
critics/<run-id>/*.weights.h5
```

The JSON is written atomically after every run. Weight files are first written under temporary names and then moved into a run-specific directory. A failed framework or seed is recorded without causing the final summary to index an empty history or overwrite artifacts from another benchmark invocation.

Open [`SLAM/SLAM-benchmark.ipynb`](SLAM/SLAM-benchmark.ipynb) after training to inspect seed-level evaluation tables, learning curves, coverage AUC, success rate, and parameter efficiency. The notebook reads schema version 3 results and contains no stale embedded outputs.

## HPC examples

The PBS examples contain no user-specific account paths:

```bash
qsub SLAM/test_vanda.pbs
qsub SLAM/train_vanda.pbs
```

Set `CONDA_ENV` or edit the resource directives for the target cluster.

## Repository layout

```text
eQMARL-SLAM-main/
├── pyproject.toml
├── environment.yml
├── requirements-classical.txt
├── requirements-quantum.txt
├── requirements-analysis.txt
├── CHANGELOG_FIXES.md
├── VALIDATION_REPORT.md
└── SLAM/
    ├── slam_core.py          # dependency-free map, FOV, transition, reward
    ├── slam.py               # Gymnasium interfaces and registrations
    ├── quantum_layers.py     # self-contained split and centralized PQCs
    ├── slam_baselines.py     # actor and four distinct critics
    ├── maa2c.py              # corrected trajectory MAA2C/GAE update
    ├── run_benchmark.py      # paired-seed train/evaluate CLI
    ├── smoke_test.py
    ├── SLAM-benchmark.ipynb
    └── tests/
```

## Results policy

No performance ranking is shipped with this corrected release. Results from the predecessor are not compatible with the new five-channel local observation, six-dimensional auxiliary state, critic interface, synchronized environment, reward scale, or corrected optimization. Retrain every framework before drawing conclusions.

## Attribution and citation

This is a modified SLAM application of the eQMARL work. Modifications are documented in `CHANGELOG_FIXES.md`. The upstream project is licensed under CC BY 4.0; the license text is retained in [`LICENSE.md`](LICENSE.md).

```bibtex
@inproceedings{derieux2025eqmarl,
  title={e{QMARL}: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels},
  author={Alexander DeRieux and Walid Saad},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=cR5GTis5II},
  doi={10.48550/arXiv.2405.17486}
}
```
