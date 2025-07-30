import os
import numpy as np
import matplotlib.pyplot as plt
import typing as t

class PlannerMetricDifferences:
    DISPLAY_NAMES = {
        "rrt-step_02-bias_02": "RRT_1",
        "rrt-step_04-bias_02": "RRT_2",
        "rrt-step_06-bias_02": "RRT_3",
        "rrt-step_02-bias_04": "RRT_4",
        "rrt-step_04-bias_04": "RRT_5",
        "rrt-step_06-bias_04": "RRT_6",
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

    def __init__(self, file_1: str, file_2: str):
        self.data_1 = np.load(file_1, allow_pickle=True)["results"]
        self.data_2 = np.load(file_2, allow_pickle=True)["results"]

    def _get_metric(self, data, planner_name: str, metric: str) -> t.Optional[float]:
        values = [
            float(entry[metric])
            for entry in data
            if entry.get("planning_successful", False)
            and self.DISPLAY_NAMES.get(entry.get("planner")) == planner_name
            and isinstance(entry.get(metric), (int, float))
        ]
        return np.mean(values) if values else None

    def plot_metric_differences(
        self,
        planner_variants: t.List[str],
        metric_to_compare: str,
        title: str,
        save_path: str,
        x_unit: str = " ",
    ):
        sim_values, real_values, labels = [], [], []

        for planner in planner_variants:
            sim_val = self._get_metric(self.data_1, planner, metric_to_compare)
            real_val = self._get_metric(self.data_2, planner, metric_to_compare)
            if sim_val is not None and real_val is not None:
                sim_values.append(sim_val)
                real_values.append(real_val)
                labels.append(planner)

        y_pos = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(sim_values, y_pos, color='skyblue', label='Simulation', marker='o', s=80)
        ax.scatter(real_values, y_pos, color='salmon', label='Real Robot', marker='x', s=80)

        for i, (sim, real) in enumerate(zip(sim_values, real_values)):
            if sim > 0:
                change_pct = 100 * (real - sim) / sim
                change_str = f"{change_pct:+.0f}%"
                max_val = max(sim, real)
                ax.text(max_val * 1.01, y_pos[i], change_str, va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.2", edgecolor='none', facecolor='lightgrey', alpha=0.8))
                ax.plot([sim, real], [y_pos[i]] * 2, color='gray', linestyle='--', linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(metric_to_compare.replace('_', ' ').title() + f" {x_unit}", fontsize=14)
        ax.set_ylabel("Planner Variant", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.invert_yaxis()
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close()


def main():
    plotter = PlannerMetricDifferences(
    file_1="/home/marcin/vm_shared/results/results_simulation_no_obstacle_with_joint_distance.npz",
    file_2="/home/marcin/vm_shared/results/results_real_robot_no_obstacle_with_joint_distance.npz"
    )

    ###########################################
    # SAVE AS NO OBSTACLE DIRECTORY (REAL ROBOT)
    ############################################
    # PLot path length differences
    plotter.plot_metric_differences(
        # RRT 1-6, RRTs 1-5, RRTc 1-4
        planner_variants=["RRT_1", "RRT_2", "RRT_3", "RRT_4", "RRT_5", "RRT_6", "RRTs_1", "RRTs_2", "RRTs_3", "RRTs_4", "RRTc_1", "RRTc_2", "RRTc_3", "RRTc_4"],
        metric_to_compare="cartesian_path_length", 
        title="Cartesian Path Length Comparison - Simulation vs Lab Hardware (Without Obstacle)",
        x_unit = "[m]",
        save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_no_obstacle/comparison_cartesian_path_length.png"
    )
    # Plot execution time differences
    plotter.plot_metric_differences(
        planner_variants=["RRT_1", "RRT_2", "RRT_3", "RRT_4", "RRT_5", "RRT_6", "RRTs_1", "RRTs_2", "RRTs_3", "RRTs_4", "RRTc_1", "RRTc_2", "RRTc_3", "RRTc_4"],
        metric_to_compare="execution_time",
        title="Execution Time Comparison - Simulation vs Lab Hardware (Without Obstacle)",
        x_unit="[s]",
        save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_no_obstacle/comparison_execution_time.png"
    )
    # Plot planning time differences
    plotter.plot_metric_differences(
        planner_variants=["RRT_1", "RRT_2", "RRT_3", "RRT_4", "RRT_5", "RRT_6", "RRTs_1", "RRTs_2", "RRTs_3", "RRTs_4", "RRTc_1", "RRTc_2", "RRTc_3", "RRTc_4"],
        metric_to_compare="planning_time",
        title="Planning Time Comparison - Simulation vs Lab Hardware (Without Obstacle)",
        x_unit="[s]",
        save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_no_obstacle/comparison_planning_time.png"
    )
    # Plot number of waypoints differences
    plotter.plot_metric_differences(
        planner_variants=["RRT_1", "RRT_2", "RRT_3", "RRT_4", "RRT_5", "RRT_6", "RRTs_1", "RRTs_2", "RRTs_3", "RRTs_4", "RRTc_1", "RRTc_2", "RRTc_3", "RRTc_4"],
        metric_to_compare="number_of_waypoints_before_post_processing",
        title="Number of Waypoints Comparison - Simulation vs Lab Hardware (Without Obstacle)",
        x_unit=" ",
        save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_no_obstacle/comparison_number_of_waypoints.png"
    )
    # plot joint travel distances differences
    plotter.plot_metric_differences(
        planner_variants=["RRT_1", "RRT_2", "RRT_3", "RRT_4", "RRT_5", "RRT_6", "RRTs_1", "RRTs_2", "RRTs_3", "RRTs_4", "RRTc_1", "RRTc_2", "RRTc_3", "RRTc_4"],
        metric_to_compare="joint_distance",
        title="Joint Travel Distances Comparison - Simulation vs Lab Hardware (Without Obstacle)",
        x_unit="[rad]",
        save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_no_obstacle/comparison_joint_travel_distances.png"
    )

if __name__ == "__main__":
    main()

# other options for metric:
      # planning_time: 22.227000000000032 (type: <class 'float'>)
      # spline_fitting_time: None (type: <class 'NoneType'>)
      # execution_time: 22.17999999999995 (type: <class 'float'>)
      # number_of_waypoints_before_post_processing: 1 (type: <class 'int'>)
      # joint_travel_distances: [0.11736239901947698 (type: <class 'list'>)
      # cartesian_path_length
