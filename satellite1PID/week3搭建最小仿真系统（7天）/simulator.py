import numpy as np
from satellite import Satellite
from controller import PIDController

def run_simulation(sat, pid, target_deg, duration, dt=0.01):
    steps = int(duration / dt)
    time = np.zeros(steps)
    angles = np.zeros(steps)
    omegas = np.zeros(steps)
    torques = np.zeros(steps)
    
    for i in range(steps):
        time[i] = i * dt
        angles[i] = sat.get_angle_deg()
        omegas[i] = sat.get_omega_deg()
        
        torque = pid.compute(target_deg, angles[i])
        torques[i] = torque
        
        sat.apply_torque(torque, dt)
    
    return {
        'time': time,
        'angle': angles,
        'omega': omegas,
        'torque': torques
    }