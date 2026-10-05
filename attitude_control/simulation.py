import numpy as np
from .core import Episode, observation
from .metrics import metrics


def rollout(config, scenario, command_fn):
    episode = Episode(config, scenario)
    angle = [np.degrees(episode.plant.theta)]
    rate = [np.degrees(episode.plant.omega)]
    torques, disturbances = [], []
    reward_sum, cost = 0.0, 0.0
    while not (episode.terminated or episode.truncated):
        command = command_fn(episode, observation(episode.plant, episode.previous_torque))
        _, reward, _, _, info = episode.step(command)
        angle.append(np.degrees(episode.plant.theta))
        rate.append(np.degrees(episode.plant.omega))
        torques.append(info["torque"])
        disturbances.append(info["disturbance"])
        reward_sum += reward
        cost += info["stage_cost_integral"]
    time = np.arange(len(angle)) * config.dt
    result = metrics(time, angle, rate, torques, config, episode.terminated)
    result.update(cost_integral=float(cost), episode_return=float(reward_sum))
    return result, {"time_s": time, "angle_deg": np.asarray(angle),
                    "rate_deg_s": np.asarray(rate), "torque_nm": np.asarray(torques),
                    "disturbance_nm": np.asarray(disturbances)}
