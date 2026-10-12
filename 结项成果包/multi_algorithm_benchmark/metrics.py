"""Shared fixed-horizon acceptance and vectorized simulation for all algorithms."""
import numpy as np
from .envs import BatchSatelliteEnv


def compute_metrics(trace, duration=20.,hold_seconds=5.,angle_tol=1.,rate_tol=.5):
    t = np.asarray(trace.get("time",trace.get("t")),float)
    error = np.asarray(trace["error_deg"],float)
    rate = np.asarray(trace["rate_deg_s"],float)
    failed = np.asarray(trace.get("failed",np.zeros_like(error,bool)),bool)
    tail = t >= duration-hold_seconds-1e-9
    in_band = (error<=angle_tol)&(rate<=rate_tol)&~failed
    complete = bool(t.size and t[-1]>=duration-1e-9)
    failures = np.flatnonzero(failed)
    failure_time = float(t[failures[0]]) if failures.size else None
    success = bool(complete and not np.any(failed) and np.any(tail) and np.all(in_band[tail]))
    exceed = np.flatnonzero(~in_band)
    start = int(exceed[-1]+1) if exceed.size else 0
    settle = float(t[start]) if success and start<len(t) and t[-1]-t[start]>=hold_seconds-1e-9 else None
    # Commands are applied on (t[k-1], t[k]]; initial row torque is zero.
    torque = np.asarray(trace["torque"],float)
    if torque.ndim == 1:
        torque = torque[:,None]
    cost = float(np.sum(np.sum(torque[1:]**2,axis=1)*np.diff(t)))
    rms = float(np.sqrt(np.trapz(error**2,t)/max(t[-1]-t[0],1e-12)))
    return {"success":success,"settling_time_s":settle,"rms_error_deg":rms,
            "tail_mean_error_deg":float(np.mean(error[tail])) if np.any(tail) else None,
            "tail_max_error_deg":float(np.max(error[tail])) if np.any(tail) else None,
            "tail_max_rate_deg_s":float(np.max(rate[tail])) if np.any(tail) else None,
            "control_cost":cost,"final_error_deg":float(error[-1]),
            "final_rate_deg_s":float(rate[-1]),"failed":bool(np.any(failed)),
            "complete":complete,"failure_time_s":failure_time,
            "active_duration_s":failure_time if failure_time is not None else float(t[-1]-t[0])}


def evaluate_batch(axis, scenarios, controller, config=None, keep_traces=True):
    env = BatchSatelliteEnv(axis,scenarios,config)
    if hasattr(controller,"reset"):
        controller.reset(env.n)
    obs = env.observation()
    shape = (env.steps+1,env.n)
    errors = np.empty(shape);rates = np.empty(shape);failures = np.empty(shape,bool)
    vector_shape = (*shape,env.dimensions)
    error_vectors = np.empty(vector_shape);omegas = np.empty(vector_shape)
    observations = np.empty((*shape,env.dimensions*2),dtype=np.float32)
    attitudes = np.empty((*shape,4 if env.axis=="three" else 1))
    torques = np.zeros(vector_shape);disturbances = np.zeros(vector_shape)
    time = np.empty(env.steps+1)
    for k in range(env.steps+1):
        time[k]=env.time;errors[k]=env.true_error_deg;rates[k]=env.rate_deg_s
        failures[k]=env.failed;error_vectors[k]=env.error_vector;omegas[k]=env.omega
        observations[k]=obs;attitudes[k]=env.q if env.axis=="three" else env.theta
        torques[k]=env.torque;disturbances[k]=env.disturbance
        if k<env.steps:
            action = controller.act(obs) if hasattr(controller,"act") else controller(obs)
            obs,_,_,_,_ = env.step(action)
    results = []
    for i,scene in enumerate(scenarios):
        trace = {"time":time.copy(),"t":time.copy(),"error_deg":errors[:,i].copy(),
                 "rate_deg_s":rates[:,i].copy(),"error_vector_rad":error_vectors[:,i].copy(),
                 "signed_error_vector":error_vectors[:,i].copy(),
                 "omega_rad_s":omegas[:,i].copy(),"torque":torques[:,i].copy(),
                 "disturbance":disturbances[:,i].copy(),"failed":failures[:,i].copy()}
        trace["observation"] = observations[:,i].copy()
        trace["quaternion" if env.axis=="three" else "theta_rad"] = attitudes[:,i].copy()
        metrics = compute_metrics(trace,duration=env.duration)
        results.append({"scenario":dict(scene),"trace":trace if keep_traces else None,"metrics":metrics})
    return results


def validation_score(results):
    metrics = [r["metrics"] for r in results]
    stats = {"success_rate":float(np.mean([m["success"] for m in metrics])),
             "rms_error_deg":float(np.mean([m["rms_error_deg"] for m in metrics])),
             "control_cost":float(np.mean([m["control_cost"] for m in metrics]))}
    score = 100*stats["success_rate"]-stats["rms_error_deg"]-.2*stats["control_cost"]
    stats["score"] = float(score)
    return float(score),stats
