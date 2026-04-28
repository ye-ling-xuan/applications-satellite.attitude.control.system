import numpy as np

class Satellite:
    def __init__(self, I=1.0):
        self.I = I
        self.theta = 0.0   # rad
        self.omega = 0.0   # rad/s
    
    def set_state(self, theta_deg, omega_deg=0.0):
        self.theta = np.radians(theta_deg)
        self.omega = np.radians(omega_deg)
    
    def apply_torque(self, torque, dt):
        alpha = torque / self.I
        self.omega += alpha * dt
        self.theta += self.omega * dt
    
    def get_angle_deg(self):
        return np.degrees(self.theta)
    
    def get_omega_deg(self):
        return np.degrees(self.omega)