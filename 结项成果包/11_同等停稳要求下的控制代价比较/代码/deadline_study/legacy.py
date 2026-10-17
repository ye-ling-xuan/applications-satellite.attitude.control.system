"""Re-evaluate existing SACs, tuning an explicit scalar on new validation only."""
from pathlib import Path
import csv
import numpy as np
from .protocol import STUDY, PACKAGE, DEADLINES, scenarios, sha256, write_json
from .evaluation import evaluate, rank

SCALES = (.25, .4, .6, .8, 1., 1.25, 1.5, 2.)


class LegacySAC:
    def __init__(self, model, axis, scale=1., row_scales=None):
        self.model, self.axis, self.scale = model, axis, float(scale)
        self.row_scales = None if row_scales is None else np.asarray(row_scales)[:, None]
        d = 1 if axis == "single" else 3
        self.zero_action = np.asarray(model.predict(np.zeros((1, 2*d), np.float32), deterministic=True)[0], float)[0]

    def reset(self, n=1):
        if self.row_scales is not None and len(self.row_scales) != n:
            raise ValueError("one action scale per row required")

    def act(self, obs, time):
        action = np.asarray(self.model.predict(np.asarray(obs, np.float32), deterministic=True)[0], float)
        calibrated = np.clip(action-self.zero_action, -1., 1.)
        return np.clip(calibrated*(self.scale if self.row_scales is None else self.row_scales), -1., 1.)


def tune_legacy():
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1)
    directory = STUDY/"03_SAC训练"/"旧模型验证与动作缩放"
    directory.mkdir(parents=True, exist_ok=True)
    rows, selected = [], []
    for axis in ("single", "three"):
        scenes = scenarios(axis, "validation")
        n = len(scenes)
        for seed in (11, 22, 33):
            path = PACKAGE/"08_惯量域随机化与泛化边界研究"/"02_模型与验证日志"/axis/"nominal"/f"seed_{seed}"/"best_model.zip"
            model = SAC.load(path, device="cpu")
            controller = LegacySAC(model, axis, row_scales=np.repeat(SCALES, n))
            metrics = evaluate(axis, scenes*len(SCALES), controller)["metrics"]
            records = []
            for i, scale in enumerate(SCALES):
                entries = metrics[i*n:(i+1)*n]
                record = {"axis": axis, "seed": seed, "action_scale": scale,
                          "validation_count": n, "mean_control_cost": float(np.mean([m["control_cost"] for m in entries]))}
                for deadline in DEADLINES:
                    record[f"passes_{int(deadline)}s"] = rank(entries, deadline)[0]
                rows.append(record)
                records.append((scale, entries, record))
            for deadline in DEADLINES:
                scale, entries, record = max(records, key=lambda value: (*rank(value[1], deadline), -abs(value[0]-1.)))
                selected.append({"kind": "legacy_sac_scaled", "axis": axis, "seed": seed,
                                 "deadline_s": deadline, "action_scale": scale,
                                 "model_path": str(path.relative_to(PACKAGE)).replace("\\", "/"),
                                 "model_sha256": sha256(path), "zero_action": controller.zero_action.tolist(),
                                 "validation": record, "fresh_training": False,
                                 "selection": "new validation; maximize passes then minimize full-horizon cost"})
            print(f"[旧SAC验证 {axis} seed={seed}] " + str({int(d): next(s for s in selected if s['axis']==axis and s['seed']==seed and s['deadline_s']==d)['action_scale'] for d in DEADLINES}), flush=True)
    with (directory/"全部动作缩放验证.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(directory/"selected_legacy.json", {"scales": list(SCALES), "split": "validation", "models": selected,
               "method": "existing zero-bias-calibrated SAC followed by fixed scalar action gain and clipping",
               "is_new_training": False, "all_original_models_retained": True})
    return selected
