from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

WORKFLOW_FILES = {
    "libero": (
        "requirements.txt",
        "env.sh",
        "setup_env.sh",
        "run_smoke.sh",
        "smoke_libero.py",
        "policy_config.yml",
        "openwam2libero_interface.py",
        "single_eval.sh",
        "single_eval.py",
        "run_eval.sh",
        "run_all_suites.py",
        "scheduler.py",
        "patches/libero-pytorch-load.patch",
    ),
    "libero-plus": (
        "environment.yml",
        "setup_env.sh",
        "run_smoke.sh",
        "smoke_libero.py",
        "policy_config.yml",
        "openwam2libero_interface.py",
        "single_eval.sh",
        "single_eval.py",
        "run_eval.sh",
        "run_all_suites.py",
        "scheduler.py",
        "patches/libero-plus-compatibility.patch",
    ),
    "robocasa365": (
        "environment.yml",
        "setup_env.sh",
        "run_smoke.sh",
        "smoke_robocasa365.py",
        "policy_config.yml",
        "openwam2robocasa365_interface.py",
        "single_eval.sh",
        "single_eval.py",
        "multi_eval.sh",
        "target_tasks.txt",
        "run_eval.sh",
    ),
}


@pytest.mark.parametrize("benchmark", sorted(WORKFLOW_FILES))
def test_complete_eval_workflow_is_kept_in_tree(benchmark: str) -> None:
    benchmark_dir = REPO_ROOT / "benchmarks" / benchmark
    missing = [name for name in WORKFLOW_FILES[benchmark] if not (benchmark_dir / name).is_file()]
    assert not missing, f"{benchmark} workflow is missing: {missing}"


@pytest.mark.parametrize(
    "relative_path",
    [
        "benchmarks/libero/setup_env.sh",
        "benchmarks/libero/run_smoke.sh",
        "benchmarks/libero/run_eval.sh",
        "benchmarks/libero-plus/setup_env.sh",
        "benchmarks/libero-plus/run_smoke.sh",
        "benchmarks/libero-plus/run_eval.sh",
        "benchmarks/robocasa365/setup_env.sh",
        "benchmarks/robocasa365/run_smoke.sh",
        "benchmarks/robocasa365/run_eval.sh",
    ],
)
def test_workflow_shell_entrypoints_are_executable(relative_path: str) -> None:
    assert os.access(REPO_ROOT / relative_path, os.X_OK)


@pytest.mark.parametrize(
    ("relative_path", "expected_name"),
    [
        ("benchmarks/libero/requirements.txt", "libero"),
        ("benchmarks/libero-plus/environment.yml", "libero-plus"),
        ("benchmarks/robocasa365/environment.yml", "robocasa365"),
    ],
)
def test_client_environment_requirements_are_valid(relative_path: str, expected_name: str) -> None:
    path = REPO_ROOT / relative_path
    if path.suffix == ".txt":
        from packaging.requirements import Requirement

        requirements = [
            Requirement(line) for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        pins = {requirement.name: str(requirement.specifier) for requirement in requirements}
        assert pins["torch"] == "==2.7.1+cpu"
        assert pins["numpy"] == "==1.22.4"
        assert pins["mujoco"] == "==3.3.2"
        return
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["name"] == expected_name
    assert payload["dependencies"]
