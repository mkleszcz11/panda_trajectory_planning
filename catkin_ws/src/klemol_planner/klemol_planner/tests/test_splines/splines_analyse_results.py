import os
import numpy as np
from klemol_planner.environment.robot_model import Robot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import typing as t
from scipy.signal import savgol_filter

# PLOTS_DIR = "/home/marcin/panda_trajectory_planning/plots"
PLOTS_DIR = "/media/sf_vm_shared"
RESULTS_FILE = "/home/marcin/panda_trajectory_planning/catkin_ws/src/klemol_planner/klemol_planner/tests/splines_results/splines_results.npz"
ENABLE_SMOOTHING = False # smooth velocity readings

def plot_3d_ee_trajectory(data: dict, waypoints_in_3d: list):
    ee = np.array(data["ee"])
    modes = np.array(data["mode"])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    mode_labels = np.unique(modes)

    # Plot end-effector paths per mode
    for mode in mode_labels:
        if mode == "undefined":
            continue
        mask = modes == mode
        ax.plot(ee[mask, 0], ee[mask, 1], ee[mask, 2], label=mode)

    # Add labeled waypoints
    for j, p in enumerate(waypoints_in_3d):
        ax.scatter(p.x, p.y, p.z, color='red', s=50)
        ax.text(p.x, p.y, p.z, f"W{j+1}", fontsize=9, color='black')

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("End-Effector Trajectory")
    ax.legend()
    ax.grid(True)

    output_path = os.path.join(PLOTS_DIR, f"trajectory_3d_merged.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_2d_ee_projections(data: dict, waypoints_3d: t.List[t.Any]):
    ee = np.array(data["ee"])
    modes = np.array(data["mode"])
    mode_labels = np.unique(modes)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    projections = [
        (0, 1, "X [m]", "Y [m]", "XY Plane"),
        (0, 2, "X [m]", "Z [m]", "XZ Plane"),
        (1, 2, "Y [m]", "Z [m]", "YZ Plane"),
    ]

    for i, (x_idx, y_idx, xlabel, ylabel, title) in enumerate(projections):
        ax = axs[i]
        for mode in mode_labels:
            if mode == "undefined":
                continue
            mask = modes == mode
            ax.plot(ee[mask, x_idx], ee[mask, y_idx], label=mode)

        # Add labeled waypoints
        for j, p in enumerate(waypoints_3d):
            x_val = getattr(p, ["x", "y", "z"][x_idx])
            y_val = getattr(p, ["x", "y", "z"][y_idx])
            ax.scatter(x_val, y_val, color='red', s=50)
            ax.text(x_val, y_val, f"W{j+1}", fontsize=9, color='black', ha='right', va='bottom')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    # os.makedirs("/mnt/data/plots", exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, f"trajectory_2d_merged.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def calculate_acceleration_jerk(time, velocity, window_length=51, polyorder=3):
    time = np.asarray(time)
    velocity = np.asarray(velocity)

    time_diffs = np.diff(time)
    valid_diffs = time_diffs[time_diffs > 0]
    delta = np.median(valid_diffs) if len(valid_diffs) > 0 else 1e-3

    acc = savgol_filter(velocity, window_length, polyorder, deriv=1, delta=delta)
    jerk = savgol_filter(velocity, window_length, polyorder, deriv=2, delta=delta)
    return acc, jerk

def plot_joint_trajectories_merged(data, robot):
    time = np.array(data["time"])
    pos = np.array(data["pos"])
    if ENABLE_SMOOTHING:
        vel = savgol_filter(np.array(data["vel"]), window_length=11, polyorder=3, axis=0)
    else:
        vel = np.array(data["vel"])
    mode = np.array(data["mode"])

    num_joints = 7
    for joint_idx in range(num_joints):
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axs[0].set_title(f"Joint {joint_idx + 1} trajectories")

        for label in np.unique(mode):
            mask = mode == label
            if np.sum(mask) < 10:
                continue

            local_time = time[mask] - time[mask][0]
            joint_pos = pos[mask, joint_idx]
            joint_vel = vel[mask, joint_idx]
            acc, jerk = calculate_acceleration_jerk(local_time, joint_vel)

            axs[0].plot(local_time, joint_pos, label=label)
            axs[1].plot(local_time, joint_vel, label=label)
            axs[2].plot(local_time, acc, label=label)
            axs[3].plot(local_time, jerk, label=label)

        for i, ylabel in enumerate(["Position [rad]", "Velocity [rad/s]", "Acceleration [rad/s²]", "Jerk [rad/s³]"]):
            axs[i].set_ylabel(ylabel)
            axs[i].grid(True)
            axs[i].legend()

        axs[3].set_xlabel("Time [s]")
        plt.tight_layout()
        filename = os.path.join(PLOTS_DIR, f"joint{joint_idx+1:02d}_merged.png")
        plt.savefig(filename, dpi=300)
        plt.close()

def plot_joint_trajectories_by_mode(results, robot):
    num_joints = 7
    for entry in results:
        mode = entry["planner"]
        time = np.array(entry["time"])
        pos = np.array(entry["pos"])
        if ENABLE_SMOOTHING:
            vel = savgol_filter(np.array(entry["vel"]), window_length=11, polyorder=3, axis=0)
        else:
            vel = np.array(entry["vel"])

        for joint_idx in range(num_joints):
            fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            axs[0].set_title(f"{mode} - Joint {joint_idx + 1}")

            joint_pos = pos[:, joint_idx]
            joint_vel = vel[:, joint_idx]
            acc, jerk = calculate_acceleration_jerk(time, joint_vel)

            axs[0].plot(time, joint_pos)
            axs[1].plot(time, joint_vel)
            axs[2].plot(time, acc)
            axs[3].plot(time, jerk)

            for i, ylabel in enumerate(["Position [rad]", "Velocity [rad/s]", "Acceleration [rad/s²]", "Jerk [rad/s³]"]):
                axs[i].set_ylabel(ylabel)
                axs[i].grid(True)

            axs[3].set_xlabel("Time [s]")
            plt.tight_layout()
            filename = os.path.join(PLOTS_DIR, f"{mode}_joint{joint_idx+1:02d}.png")
            plt.savefig(filename, dpi=300)
            plt.close()

def plot_joint_trajectories_all_on_separate_plots(results, robot):
    num_joints = 7
    param_labels = ["Position [rad]", "Velocity [rad/s]", "Acceleration [rad/s²]", "Jerk [rad/s³]"]
    param_colors = ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3"]  # print-safe

    for joint_idx in range(num_joints):
        fig, axs = plt.subplots(len(results), 4, figsize=(20, 3.5 * len(results)), sharex='col')

        for row_idx, entry in enumerate(results):
            mode = entry["planner"]
            time = np.array(entry["time"]) - entry["time"][0]
            pos = np.array(entry["pos"])[:, joint_idx]
            vel = np.array(entry["vel"])[:, joint_idx]
            acc, jerk = calculate_acceleration_jerk(time, vel)

            joint_data = [pos, vel, acc, jerk]

            for col_idx, (data_series, ylabel, color) in enumerate(zip(joint_data, param_labels, param_colors)):
                ax = axs[row_idx, col_idx] if len(results) > 1 else axs[col_idx]
                ax.plot(time, data_series, color=color, linewidth=1.5)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.grid(True)
                if row_idx == len(results) - 1:
                    ax.set_xlabel("Time [s]", fontsize=10)
                if row_idx == 0:
                    ax.set_title(ylabel, fontsize=11, pad=10)

        # Add planner labels outside the plot area on the left
        for row_idx, entry in enumerate(results):
            fig.text(0.03, 1 - (row_idx + 0.5) / len(results), entry["planner"],
                     va='center', ha='right', fontsize=12, rotation=90)

        fig.suptitle(f"Trajectories – Joint {joint_idx + 1}", fontsize=16, y=0.995)
        plt.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08, wspace=0.3, hspace=0.25)
        plot_path = os.path.join(PLOTS_DIR, f"joint{joint_idx + 1:02d}_grid.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

def plot_all_joint_trajectories_all_splines_per_one_parameter(results, robot):
    num_joints = 7
    param_labels = ["Position [rad]", "Velocity [rad/s]", "Acceleration [rad/s²]", "Jerk [rad/s³]"]
    param_colors = ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3"]  # print-safe

    for param_idx, (param_label, color) in enumerate(zip(param_labels, param_colors)):
        fig, axs = plt.subplots(len(results), num_joints, figsize=(4 * num_joints, 3.5 * len(results)), sharex='col')

        if len(results) == 1:
            axs = np.expand_dims(axs, axis=0)
        if num_joints == 1:
            axs = np.expand_dims(axs, axis=1)

        for row_idx, entry in enumerate(results):
            mode = entry["planner"]
            time = np.array(entry["time"]) - entry["time"][0]
            pos = np.array(entry["pos"])
            vel = np.array(entry["vel"])

            for joint_idx in range(num_joints):
                joint_pos = pos[:, joint_idx]
                joint_vel = vel[:, joint_idx]
                acc, jerk = calculate_acceleration_jerk(time, joint_vel)
                joint_data = [joint_pos, joint_vel, acc, jerk][param_idx]

                ax = axs[row_idx, joint_idx]
                ax.plot(time, joint_data, color=color, linewidth=1.5)
                ax.grid(True)

                if row_idx == len(results) - 1:
                    ax.set_xlabel("Time [s]", fontsize=9)
                if row_idx == 0:
                    ax.set_title(f"J{joint_idx + 1}", fontsize=10, pad=10)

        # Planner names on left
        for row_idx, entry in enumerate(results):
            fig.text(0.02, 1 - (row_idx + 0.5) / len(results), entry["planner"],
                     va='center', ha='right', fontsize=11, rotation=90)

        # Y-label on whole figure
        fig.text(0.005, 0.5, param_label, va='center', ha='center',
                 rotation='vertical', fontsize=12)

        fig.suptitle(f"Joint {param_label} Trajectories", fontsize=14, y=0.995)
        plt.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08, wspace=0.3, hspace=0.3)

        plot_path = os.path.join(PLOTS_DIR, f"trajectories_param_{param_label.split()[0].lower()}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

def plot_joint_trajectories_all_splines_per_one_parameter(results, robot):
    num_joints = 7
    param_labels = ["Position [rad]", "Velocity [rad/s]", "Acceleration [rad/s²]", "Jerk [rad/s³]"]
    param_colors = ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3"]

    for joint_idx in range(num_joints):
        for param_idx, (param_label, color) in enumerate(zip(param_labels, param_colors)):
            fig, axs = plt.subplots(1, len(results), figsize=(6 * len(results), 4), sharey=True)

            if len(results) == 1:
                axs = [axs]  # ensure iterable

            for col_idx, entry in enumerate(results):
                mode = entry["planner"]
                time = np.array(entry["time"]) - entry["time"][0]
                pos = np.array(entry["pos"])[:, joint_idx]
                vel = np.array(entry["vel"])[:, joint_idx]
                acc, jerk = calculate_acceleration_jerk(time, vel)
                joint_data = [pos, vel, acc, jerk][param_idx]

                ax = axs[col_idx]
                ax.plot(time, joint_data, color=color, linewidth=1.5)
                ax.set_title(mode, fontsize=11)
                ax.set_xlabel("Time [s]")
                ax.grid(True)

                if col_idx == 0:
                    ax.set_ylabel(param_label)

            fig.suptitle(f"{param_label} – Joint {joint_idx + 1}", fontsize=14, y=1.02)
            plt.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.2, wspace=0.3)

            filename = os.path.join(
                PLOTS_DIR,
                f"joint{joint_idx + 1:02d}_{param_label.split()[0].lower()}_sidebyside.png"
            )
            plt.savefig(filename, dpi=300)
            plt.close()


def main():
    # Create output directory if not exists
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load results from file
    data = np.load(RESULTS_FILE, allow_pickle=True)
    results = data["results"]
    robot = Robot()

    # Print planner performance
    for entry in results:
        planner = entry.get("planner", "unknown")
        planning_time = entry.get("planning_time")
        execution_time = entry.get("execution_time")
        print(f"\nPlanner: {planner}")
        print(f"  Planning Time:   {planning_time:.3f} s" if planning_time else "  Planning Time:   not recorded")
        print(f"  Execution Time:  {execution_time:.3f} s" if execution_time else "  Execution Time:  not recorded")

    # Merge data across all planners for joint/EE plots
    merged_data = {
        "time": np.concatenate([r["time"] for r in results]),
        "pos": np.concatenate([r["pos"] for r in results]),
        "vel": np.concatenate([r["vel"] for r in results]),
        "ee":  np.concatenate([r["ee"] for r in results]),
        "mode": np.concatenate([
            np.array([r["planner"]] * len(r["time"])) for r in results
        ])
    }

    # --- Call plotting functions ---
    waypoints = [
        np.array([0.0, -0.786, 0.0, -2.356, 0.0, 1.572, 0.785]),
        np.array([0.682, 0.025, -0.038, -2.118, -0.002, 2.145, 1.507]),
        np.array([0.579, 0.303, 0.062, -2.237, -0.038, 2.542, 1.533]),
        # np.array([0.152, 0.415, 0.549, -2.224, -0.388, 2.549, 1.774]),
        np.array([-0.456, 0.084, 0.498, -1.988, -0.042, 2.065, 0.858]), #np.array([-0.982, 0.356, 1.240, -2.018, -0.394, 2.105, 1.231]),
        np.array([-0.542, -0.626, 0.055, -2.502, 0.831, 2.165, 0.816]),
        np.array([-0.261, -0.068, -0.534, -2.334, -0.117, 2.260, -0.060])

    ]

    waypoints_in_3d = [robot.fk(q) for q in waypoints]
    print("Waypoints in 3D:", waypoints_in_3d)

    plot_3d_ee_trajectory(merged_data, waypoints_in_3d)
    print("3D EE trajectory plot saved.")

    plot_2d_ee_projections(merged_data, waypoints_in_3d)
    print("2D EE projections plot saved.")

    plot_joint_trajectories_merged(merged_data, robot)
    print("Merged joint trajectories plots saved.")

    plot_joint_trajectories_by_mode(results, robot)
    print("Joint trajectories plots saved.")

    plot_joint_trajectories_all_on_separate_plots(results, robot)
    print("Joint trajectories on separate plots saved.")

    plot_all_joint_trajectories_all_splines_per_one_parameter(results, robot)
    print("Joint trajectories on separate plots v2 saved.")

    plot_joint_trajectories_all_splines_per_one_parameter(results, robot)
    print("Joint trajectories on separate plots v3 saved.")


if __name__ == "__main__":
    main()