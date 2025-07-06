import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def calculate_acceleration_jerk(time, velocity, window_length=51, polyorder=3):
    # First, calculate acceleration by differentiating velocity
    acc = savgol_filter(velocity, window_length, polyorder, deriv=1, delta=(time[1] - time[0]))
    
    # Then, calculate jerk by differentiating acceleration
    jerk = savgol_filter(velocity, window_length, polyorder, deriv=2, delta=(time[1] - time[0]))

    return acc, jerk

def plot_joint_comparison(data, joint_index=0, velocity_cap=2.6100 * 0.25):
    time = data['time']
    positions = data['pos']
    velocities = data['vel']
    modes = data['mode']

    pos = np.array(positions)[:, joint_index]
    vel = np.array(velocities)[:, joint_index]

    # # Cap velocities to avoid unrealistic spikes
    # vel = np.clip(vel, -velocity_cap, velocity_cap)

    # Calculate acceleration and jerk using Savitzky-Golay filter
    acc, jerk = calculate_acceleration_jerk(time, vel)

    mode_labels = np.unique(modes)

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Plot each mode on top of each other, aligned to time 0
    for mode in mode_labels:
        if mode == "undefined":
            continue
        mask = modes == mode
        mode_time = time[mask] - time[mask][0]  # Align time to start from 0
        
        axs[0].plot(mode_time, pos[mask], label=f"Position ({mode})")
        axs[1].plot(mode_time, vel[mask], label=f"Velocity ({mode})")
        axs[2].plot(mode_time, acc[mask], label=f"Acceleration ({mode})")
        axs[3].plot(mode_time, jerk[mask], label=f"Jerk ({mode})")

    axs[0].set_ylabel("Position [rad]")
    axs[1].set_ylabel("Velocity [rad/s]")
    axs[2].set_ylabel("Acceleration [rad/s²]")
    axs[3].set_ylabel("Jerk [rad/s³]")
    axs[3].set_xlabel("Time [s]")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    plt.suptitle(f"Joint {joint_index + 1} Trajectory Analysis")
    plt.tight_layout()

def plot_all_joints(data):
    for i in range(7):  # Plot for all joints
        plot_joint_comparison(data, joint_index=i)
    plt.show()  # Show all the figures at once

if __name__ == "__main__":
    file_path = "/tmp/pick_banana_demo.npz"
    data = np.load(file_path)

    plot_all_joints(data)