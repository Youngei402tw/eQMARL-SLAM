# Validation report

Validation date: **2026-07-22**

This report distinguishes checks completed in the available execution environment from checks that require the repository's pinned legacy TensorFlow Quantum stack.

## Validation environment

- Host interpreter: Python 3.13.5
- Available for direct execution: NumPy, pytest, Jupyter/nbclient, standalone Keras 3 with the JAX backend
- Unavailable in the host interpreter: Gymnasium, TensorFlow 2.7.0, Cirq 0.13.1, TensorFlow Quantum 0.7.2
- Target runtime declared by the project: Python 3.9 with TensorFlow 2.7.0 and TensorFlow Quantum 0.7.2

The target stack could not be provisioned in this container because a Python 3.9 runtime and compatible binary packages were not locally available, and the package index was unavailable. Consequently, no claim is made that a real TFQ training job completed here.

## Completed checks

### 1. Python parsing and package build

```text
python -m compileall -q SLAM
python -m pip wheel . --no-deps --ignore-requires-python --no-build-isolation
```

All Python modules parsed successfully. The wheel build completed and contained the package modules plus the `eqmarl-slam-benchmark` console entry point.

### 2. Automated tests

```text
30 passed, 3 skipped
```

The skipped modules are dependency-gated checks for Gymnasium, TensorFlow, and TensorFlow Quantum. The 30 executed checks cover:

- deterministic seeded map generation;
- connected free-space maps for empty, hard, and random generators;
- absence of ground-truth map leakage;
- consistent `[y, x]` array indexing;
- distinct per-agent local belief maps;
- synchronous collision and movement resolution;
- no collision penalty for valid turns;
- bounded, map-normalized rewards;
- reachable/observable coverage termination;
- semantic binary observation channels;
- episode lifecycle enforcement;
- integer action validation;
- finite reward-coefficient validation;
- start-position validation;
- removal of the missing submodule and stale result artifacts;
- source-level exclusion of the predecessor's known bad patterns;
- absence of stale notebook outputs and presence of release documentation/dependency pins.

### 3. Randomized environment invariants

A property-style randomized run covered:

```text
360 rollouts
9,335 joint transitions
map sizes: 6, 8, 16, 32
maze types: empty, hard, random
agent counts: 1, 2, 4
```

The following invariants held in every sampled transition:

- free space remained connected;
- observations retained the declared shape and semantics;
- coverage never decreased;
- agent positions remained unique;
- every FOV cell belonged to the observable target mask;
- hidden walls were not disclosed through local observations;
- rewards remained finite.

### 4. Classical model graph/forward validation

Because TensorFlow 2.7 was unavailable, the actor, sCTDE critic, and fCTDE critic were additionally instantiated through standalone Keras 3 using the JAX backend as a structural compatibility check. Forward passes succeeded with these shapes:

```text
actor: (2, 3)
sCTDE critic: (3, 1)
fCTDE critic: (3, 1)
```

The actor probabilities summed to one. The two classical critics had different graphs and parameter counts:

```text
actor: 18,371 parameters
sCTDE critic: 28,801 parameters
fCTDE critic: 91,905 parameters
```

This check is supplementary; it is not a substitute for TensorFlow 2.7 execution.

### 5. Notebook execution

`SLAM/SLAM-benchmark.ipynb` was executed from beginning to end against a synthetic schema-version-3 benchmark result containing multiple frameworks and model seeds. All 15 cells completed without an error. The release notebook itself was then retained with empty outputs and null execution counts.

## Dependency-gated checks to run in the target environment

After creating the Python 3.9 environment described in `environment.yml`, run:

```bash
python -m pytest
python -m SLAM.smoke_test
python -m SLAM.run_benchmark \
  --quick \
  --env-id MiniGrid-RandomSLAM-6x6-v0 \
  --frameworks eqmarl qfctde sctde fctde \
  --strict-quantum \
  --no-progress
```

In that environment, the previously skipped tests exercise:

- Gymnasium's environment checker and registered IDs;
- TensorFlow actor and classical critic forward passes;
- the GAE terminal/truncation calculation;
- Cirq/TFQ quantum-layer construction and forward passes;
- matched eQMARL/qfCTDE readout dimensions.

## Scientific scope

The corrected simulator supplies exact agent position and heading. It is therefore a cooperative mapping and active-exploration benchmark under known localization. It does not model pose uncertainty, scan matching, loop closure, or graph-based SLAM optimization. Performance claims should use newly trained held-out evaluation results; predecessor JSON and weights are intentionally excluded.


## 2.0.1 sampling hotfix validation

A real GPU run exposed a dtype-boundary defect that the original smoke tests did
not exercise: TensorFlow float32 softmax rows were cast to NumPy float64 without
being renormalized at float64 precision. `Generator.choice` then rejected some
valid distributions with `ValueError: probabilities do not sum to 1`.

Version 2.0.1 converts, clips, and normalizes the sampling rows in NumPy float64
and absorbs the residual into the largest action probability. A pure NumPy
regression test reproduces the exact `[0.33333334, 0.33333334, 0.33333334]`
case and performs repeated categorical draws. The local validation result is:

```text
32 passed, 3 skipped
```

The skipped groups still require Gymnasium, TensorFlow, and TensorFlow Quantum in
the validation container. The numerical regression itself is not skipped.
