import numpy as np
import matplotlib.pyplot as plt
import os


DISPLAY_NAMES = {
    # "rrt-step_02-bias_02": "RRT_1",
    # "rrt-step_04-bias_02": "RRT_2",
    # "rrt-step_06-bias_02": "RRT_3",
    # "rrt-step_02-bias_04": "RRT_4",
    # "rrt-step_04-bias_04": "RRT_5",
    # "rrt-step_06-bias_04": "RRT_6",
    "rrt_star-step_02-bias_02-rewire_061": "RRTs_1",
    "rrt_star-step_04-bias_02-rewire_081": "RRTs_2",
    "rrt_star-step_06-bias_02-rewire_121": "RRTs_3",
    "rrt_star-step_04-bias_04-rewire_081": "RRTs_4",
    "rrt_star-step_06-bias_04-rewire_121": "RRTs_5",
    "rrt_with_connecting-step_02-bias_02": "RRTc_1",
    "rrt_with_connecting-step_04-bias_02": "RRTc_2",
    "rrt_with_connecting-step_01-bias_01": "RRTc_3",
    "rrt_with_connecting-step_06-bias_06": "RRTc_4",
    "prm_1000_samples_not_restricted": "PRM_1",
    "prm_10000_samples_not_restricted": "PRM_2",
    "prm_100000_samples_not_restricted": "PRM_3",
    "prm_1000_samples_not_restricted_diff_w": "PRM_4",
    "prm_10000_samples_not_restricted_diff_w": "PRM_5",
    "prm_100000_samples_not_restricted_diff_w": "PRM_6",
    "prm_1000_samples_not_restricted_noc": "PRM_7",
    "prm_10000_samples_not_restricted_noc": "PRM_8",
    "prm_100000_samples_not_restricted_noc": "PRM_9",
    "prm_1000_diff_w_samples_restricted_noc": "PRM_10",
    "prm_10000_diff_w_samples_restricted_noc": "PRM_11",
    "prm_100000_diff_w_samples_restricted_noc": "PRM_12",
    "prm_10000_samples_restricted_fixed_tool": "PRM_13",
    "moveit_default": "MoveIt",
}

MOVEIT_KEY = "moveit_default"

def get_total_time(entry: dict) -> float:
    try:
        p = float(entry.get("planning_time", 0) or 0)
        s = float(entry.get("spline_fitting_time", 0) or 0)
        e = float(entry.get("execution_time", 0) or 0)
        return p + s + e
    except Exception:
        return None

def plot_moveit_vs_custom(npz_path: str, save_path: str):
    data = np.load(npz_path, allow_pickle=True)["results"]

    planner_to_times = {}
    for entry in data:
        if not entry.get("planning_successful", False):
            continue
        key = entry.get("planner")
        if key not in DISPLAY_NAMES:
            continue
        name = DISPLAY_NAMES[key]

        if name == "MoveIt":
            val = entry.get("execution_time")
        else:
            val = get_total_time(entry)

        if val is None:
            continue

        planner_to_times.setdefault(name, []).append(val)

    # Compute average times
    averaged_times = {k: np.mean(v) for k, v in planner_to_times.items()}

    # Skip if MoveIt is missing
    if "MoveIt" not in averaged_times:
        print("No MoveIt data found.")
        return

    moveit_time = averaged_times["MoveIt"]
    planners = [p for p in averaged_times if p != "MoveIt"]
    planners.sort()

    # Setup figure
    fig_height = max(2.5, 0.35 * len(planners))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_ticks = []
    y_labels = []

    for idx, planner in enumerate(planners):
        y_ticks.append(idx)
        y_labels.append(planner)

        impl_time = averaged_times[planner]

        # Horizontal line and markers
        ax.plot([moveit_time, impl_time], [idx, idx], 'k--', alpha=0.4)
        ax.plot(moveit_time, idx, 'o', color="#648fff", label="MoveIt" if idx == 0 else "")
        ax.plot(impl_time, idx, 'x', color="#dc267f", markersize=10, label="Custom Planner" if idx == 0 else "")

        # Annotate with % difference
        delta = (impl_time - moveit_time) / moveit_time * 100
        ax.text(
            max(moveit_time, impl_time) + 0.04,
            idx,
            f"{delta:+.1f}%",
            va='center',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='none', facecolor='lightgrey', alpha=0.8)
        )

    # Axis config
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel("Total Execution Time [s]", fontsize=14)
    ax.set_title("Full Execution Time - MoveIt vs. Implemented Planners (Obstacle)", fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(True, axis='x', linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=12, title="Legend", title_fontsize=12)

    # Layout and save
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    plot_moveit_vs_custom(
        npz_path="/home/marcin/vm_shared/results/results_real_robot_obstacle.npz",
        save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_obstacle/00_moveit_vs_custom_time.png"
    )