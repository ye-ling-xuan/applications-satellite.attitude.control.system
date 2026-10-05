from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
from .config import Config
from .core import Scenario, DYNAMICS_VERSION, OBSERVATION_VERSION, REWARD_VERSION


def create_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}; choose a new directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def contract(config):
    return {"config": config.to_dict(), "dynamics": DYNAMICS_VERSION,
            "observation": OBSERVATION_VERSION, "reward": REWARD_VERSION}


def check_contract(metadata, config, mode):
    if metadata.get("mode") != mode:
        raise ValueError("model control mode mismatch")
    if metadata.get("contract") != contract(config):
        raise ValueError("model and evaluation physics/observation/controller config mismatch")


def provenance():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "source_sha256": source_hashes()}


def load_config(tuning=None):
    if tuning is None:
        return Config()
    data = json.loads(Path(tuning).read_text(encoding="utf-8"))
    return Config(**data["pd_config"])


def scenario_suite(config, seed, angles=None):
    """Held-out suite: identical initial states and disturbance realizations for all methods."""
    rng = np.random.default_rng(seed)
    angles = angles or [-60, -30, -10, 10, 30, 60]
    groups = [
        ("nominal", config.inertia, 0.0, 0.0, 0.0),
        ("low_inertia", 0.5 * config.inertia, 0.0, 0.0, 0.0),
        ("high_inertia", 1.5 * config.inertia, 0.0, 0.0, 0.0),
        ("bias_positive", config.inertia, 0.08, 0.0, 0.0),
        ("bias_negative", config.inertia, -0.08, 0.0, 0.0),
        ("pulse_noise", config.inertia, 0.0, 0.3, 0.01),
    ]
    for group, inertia, bias, pulse, noise in groups:
        for index, angle in enumerate(angles):
            yield group, Scenario(name=f"{group}_{index}", angle_deg=angle,
                                  rate_deg_s=float(rng.uniform(-10, 10)), inertia=inertia,
                                  bias_torque=bias, pulse_torque=pulse, noise_std=noise,
                                  seed=int(rng.integers(0, 2**31)))


def scenario_dict(scenario):
    return asdict(scenario)
