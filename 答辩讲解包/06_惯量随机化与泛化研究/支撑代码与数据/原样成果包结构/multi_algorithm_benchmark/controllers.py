"""Controllers use only the same measured error and angular-rate observations."""
from pathlib import Path
import csv
import json
import numpy as np
from .envs import axis_name


class ModelController:
    def __init__(self, model, axis, physical_action=False, max_torque=None):
        self.model = model
        self.axis = axis_name(axis)
        self.physical_action = physical_action
        self.limit = float(max_torque) if max_torque is not None else (.5 if self.axis == "single" else 2.)

    def reset(self, n=1):
        self.n = n

    def act(self, observations):
        actions,_ = self.model.predict(np.asarray(observations,np.float32),deterministic=True)
        actions = np.asarray(actions,float)
        if self.physical_action:
            # Historical archives store physical N m actions; conversion keeps
            # their physical request, then the shared deployment actuator clips.
            actions = actions/self.limit
        return np.clip(actions,-1,1)


class PIDController:
    def __init__(self, axis, kind="tuned", gains=None):
        self.axis = axis_name(axis)
        self.dimensions = 1 if self.axis == "single" else 3
        if kind == "baseline":
            kind = "legacy"
        if kind not in ("legacy","tuned","improved"):
            raise ValueError("unknown PID kind")
        self.kind = kind
        self.dt = .05 if self.axis == "single" else .02
        self.limit = .5 if self.axis == "single" else 2.
        defaults = dict(kp=4.,ki=0.,kd=3.,alpha=1.,integral_limit=2.) if self.axis == "single" else dict(kp=3.,ki=.5,kd=1.,alpha=1.,integral_limit=5.)
        self.gains = {**defaults, **dict(gains or {})}
        for key in ("kp","ki","kd","integral_limit"):
            if not np.isfinite(self.gains[key]) or self.gains[key] < 0:
                raise ValueError("PID gains must be finite and nonnegative")
        if not 0 < self.gains["alpha"] <= 1:
            raise ValueError("filter alpha must be in (0,1]")
        self.reset(1)

    def reset(self, n=1):
        self.integral = np.zeros((n,self.dimensions))
        self.previous = np.zeros_like(self.integral)
        self.filtered_rate = np.zeros_like(self.integral)
        self.first = True

    def act(self, observation):
        obs = np.asarray(observation,float)
        one = obs.ndim == 1
        obs = obs.reshape(-1,self.dimensions*2)
        if len(obs) != len(self.integral):
            self.reset(len(obs))
        error,rate = obs[:,:self.dimensions],obs[:,self.dimensions:]
        # Three-axis observation is desired correction; single observation is
        # current minus target. The correction signs are therefore opposite.
        correction = -error if self.axis == "single" else error
        g = self.gains
        candidate = np.clip(self.integral+correction*self.dt,-g["integral_limit"],g["integral_limit"])
        if self.kind == "legacy" and self.axis == "three":
            derivative = (correction-self.previous)/self.dt
            self.integral = candidate
            torque = g["kp"]*correction+g["ki"]*self.integral+g["kd"]*derivative
        else:
            alpha = g["alpha"] if self.kind == "improved" else 1.
            if self.first:
                self.filtered_rate = rate.copy()
            else:
                self.filtered_rate += alpha*(rate-self.filtered_rate)
            base = g["kp"]*correction-g["kd"]*self.filtered_rate
            proposed = base+g["ki"]*candidate
            delta = g["ki"]*(candidate-self.integral)
            accept = (np.abs(proposed)<=self.limit)|((proposed>self.limit)&(delta<0))|((proposed<-self.limit)&(delta>0))
            self.integral = np.where(accept,candidate,self.integral)
            torque = base+g["ki"]*self.integral
        self.previous = correction.copy()
        self.first = False
        result = np.clip(torque,-self.limit,self.limit)/self.limit
        return result[0] if one else result


def tune_pid(axis, output_dir):
    """Freeze gains using validation only; preserve every candidate's evidence."""
    from .metrics import evaluate_batch, validation_score
    from .scenarios import validation_scenarios
    axis = axis_name(axis)
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    scenes = validation_scenarios(axis)
    default = PIDController(axis,"legacy").gains
    selected = {"legacy":default}
    rows = []
    # One common P/D grid; each family gets the same Ki candidates. Filtering
    # adds a declared alpha search, rather than accessing privileged true rates.
    kp_grid = (2.,4.,8.,12.) if axis == "single" else (2.,4.,8.,12.)
    kd_grid = (1.5,3.,5.) if axis == "single" else (2.,4.,6.)
    candidates = [(kp,ki,kd) for kp in kp_grid for ki in (0.,.08,.2,.5) for kd in kd_grid]
    for kind in ("tuned","improved"):
        best = None
        alphas = (1.,) if kind == "tuned" else (.2,.5,1.)
        for kp,ki,kd in candidates:
            for alpha in alphas:
                gains = {**default,"kp":kp,"ki":ki,"kd":kd,"alpha":alpha}
                result = evaluate_batch(axis,scenes,PIDController(axis,kind,gains),keep_traces=False)
                score,stats = validation_score(result)
                rows.append({"axis":axis,"kind":kind,**gains,**stats})
                # Deterministic tie break prefers lower RMS then lower cost.
                rank = (score,-stats["rms_error_deg"],-stats["control_cost"])
                if best is None or rank > best[0]:
                    best = (rank,gains)
        selected[kind] = best[1]
    path = output/"pid_configs.json"
    all_configs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    all_configs[axis] = selected
    path.write_text(json.dumps(all_configs,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output/f"pid_search_{axis}.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return selected
