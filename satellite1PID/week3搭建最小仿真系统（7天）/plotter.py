import matplotlib.pyplot as plt

def plot_results(data):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].plot(data['time'], data['angle'])
    axes[0].set_ylabel('Angle (deg)')
    axes[0].grid(True)
    
    axes[1].plot(data['time'], data['omega'])
    axes[1].set_ylabel('Omega (deg/s)')
    axes[1].grid(True)
    
    axes[2].plot(data['time'], data['torque'])
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Torque (Nm)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()