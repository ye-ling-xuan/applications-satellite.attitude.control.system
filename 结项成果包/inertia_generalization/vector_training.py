"""Pure-inertia reset distribution; unchanged V2 vectorized RK4 propagation."""
from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

from multi_algorithm_benchmark.vector_training import FastThreeVecEnv as _FastThreeVecEnv
from .envs import EpisodeSampler, InertiaSatelliteEnv, mode_name


class FastThreeVecEnv(_FastThreeVecEnv):
    """V3 sampler plus the fully verified V2 SB3/Monitor/auto-reset engine.

    Initial states from all resets are written, including short failed episodes
    and automatically restarted rows. The log describes actual training draws,
    not a promised equality of episode counts between different learned policies.
    """
    def __init__(self, n_envs=8, seed=0, monitor_dir=None, mode="nominal", config=None):
        self.mode = mode_name(mode)
        self._samplers = [EpisodeSampler("three", self.mode, int(seed)+rank) for rank in range(n_envs)]
        self._episode_record_stream = None
        self._episode_record_writer = None
        super().__init__(n_envs, seed, monitor_dir, randomized=False, config=config)
        if monitor_dir is not None:
            path = Path(monitor_dir) / "training_episode_initial_states.csv"
            self._episode_record_stream = path.open("x", newline="", encoding="utf-8-sig")
            self._episode_record_writer = csv.DictWriter(self._episode_record_stream,
                fieldnames=["rank", "episode_index", "stream_seed", "mode", "euler_deg",
                            "omega_deg_s", "inertia_scale", "noise_seed"])
            self._episode_record_writer.writeheader()

    def config_dict(self):
        return InertiaSatelliteEnv("three", self.mode, self.config).config_dict()

    def _reset_row(self, rank, seed=None, options=None):
        if seed is not None:
            self._samplers[rank].reseed(seed)
        scene, record = self._samplers[rank].sample(options)
        # Parent's unused historical reset draws cannot affect any V3 stream;
        # all physical fields are overridden by our explicit pure-inertia scene.
        super()._reset_row(rank, seed=seed, options=scene)
        self.reset_infos[rank].update(inertia_scale=deepcopy(scene["inertia_scale"]),
                                      episode_initial_state=record)
        if self._episode_record_writer:
            row = {key: record[key] for key in ("episode_index", "stream_seed", "mode", "noise_seed")}
            row.update(rank=rank, euler_deg=json.dumps(scene["euler_deg"]),
                       omega_deg_s=json.dumps(scene["omega_deg_s"]),
                       inertia_scale=json.dumps(scene["inertia_scale"]))
            self._episode_record_writer.writerow(row)
            self._episode_record_stream.flush()

    def close(self):
        if self._episode_record_stream is not None and not self._episode_record_stream.closed:
            self._episode_record_stream.close()
        super().close()


InertiaThreeVecEnv = FastThreeVecEnv
