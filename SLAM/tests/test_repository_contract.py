from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_removed_submodule_and_stale_result_artifacts() -> None:
    assert not (ROOT / ".gitmodules").exists()
    assert not (ROOT / "eqmarl").exists()
    generated_files = [
        path
        for path in (ROOT / "SLAM" / "results").rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    assert generated_files == []


def test_known_bad_code_patterns_are_absent() -> None:
    source_files = [
        ROOT / "SLAM" / "slam_core.py",
        ROOT / "SLAM" / "slam.py",
        ROOT / "SLAM" / "maa2c.py",
        ROOT / "SLAM" / "slam_baselines.py",
        ROOT / "SLAM" / "run_benchmark.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    for forbidden in (
        "self.grid.encode",
        " / 255.0",
        "eqmarl.algorithms",
        "alpha = 0.05",
        "policy_loss + self.entropy",
        "maze_type=\"random\"  # fixed",
    ):
        assert forbidden not in source
    assert "actor_loss = policy_loss - self.entropy_coef * entropy" in source
    assert "tf.clip_by_global_norm" in source
    assert "training=False)," in source


def test_notebook_contains_no_stale_outputs() -> None:
    import json

    notebook = json.loads((ROOT / "SLAM" / "SLAM-benchmark.ipynb").read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None


def test_release_documents_and_dependency_pins_are_present() -> None:
    assert (ROOT / "VALIDATION_REPORT.md").is_file()
    assert (ROOT / "CHANGELOG_FIXES.md").is_file()
    quantum_requirements = (ROOT / "requirements-quantum.txt").read_text()
    for requirement in (
        "tensorflow-quantum==0.7.2",
        "cirq-core==0.13.1",
        "sympy==1.8",
        "protobuf==3.17.3",
    ):
        assert requirement in quantum_requirements
