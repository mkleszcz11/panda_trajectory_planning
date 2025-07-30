import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import Patch


class PlannerAnalysis:
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

    PLANNER_GROUP = {
        "RRT": ["RRT_1", "RRT_2", "RRT_3", "RRT_4", "RRT_5", "RRT_6"],
        "RRT*": ["RRTs_1", "RRTs_2", "RRTs_3", "RRTs_4", "RRTs_5"],
        "RRT-Connect": ["RRTc_1", "RRTc_2", "RRTc_3", "RRTc_4"],
        "PRM": ["PRM_" + str(i) for i in range(1, 14)],
        "MoveIt": ["MoveIt"],
    }

    GROUP_COLORS = {
        "RRT": '#ffb000',
        "RRT*": '#785ef0',
        "RRT-Connect": '#dc267f',
        "PRM": '#fe6100',
        "MoveIt": '#648fff',
        "Other": '#44aa99',
    }

    def __init__(self, npz_path: str):
        self.data = np.load(npz_path, allow_pickle=True)["results"]

    def get_group(self, display_name: str) -> str:
        for group, names in self.PLANNER_GROUP.items():
            if display_name in names:
                return group
        return "Other"

    def _extract_data_for_metric(self, metric_name: str):
        grouped_data = {}
        planner_order = []

        for entry in self.data:
            planner_key = entry.get("planner")
            if planner_key not in self.DISPLAY_NAMES:
                continue

            name = self.DISPLAY_NAMES[planner_key]
            value = entry.get(metric_name)

            # Filter out missing, None, NaN, or non-numeric
            if value is None or isinstance(value, str) or value < 0.01:
                continue
            try:
                val = float(value)
            except (ValueError, TypeError):
                continue

            if name not in grouped_data:
                grouped_data[name] = []
                planner_order.append(name)

            grouped_data[name].append(val)

        # Remove planners with empty data
        grouped_data = {k: v for k, v in grouped_data.items() if len(v) > 0}
        planner_order = [k for k in planner_order if k in grouped_data]

        return grouped_data, planner_order

    def _plot_metric(self, grouped_data, planner_order, ylabel, title, save_path):
        data = [grouped_data[name] for name in planner_order]
        colors = [self.GROUP_COLORS[self.get_group(name)] for name in planner_order]

        plt.rcParams.update({
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
        })

        fig, ax = plt.subplots(figsize=(14, 4))
        bp = ax.boxplot(data, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)

        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        ax.set_xticklabels(planner_order, rotation=45)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

        legend_elements = [
            Patch(facecolor=color, edgecolor='black', label=group)
            for group, color in self.GROUP_COLORS.items()
            if any(self.get_group(p) == group for p in planner_order)
        ]
        ax.legend(
            handles=legend_elements,
            title="Planner Type",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            frameon=True
        )

        plt.tight_layout(rect=[0, 0, 1, 1])
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

            plt.close()
        else:
            plt.show()

    def plot_cartesian_path_length(self, save_path=None, plot_title = "Cartesian Path Length - TODO, TODO"):
        data, order = self._extract_data_for_metric("cartesian_path_length")
        self._plot_metric(data, order, "Cartesian Path Length [m]", plot_title, save_path)

    def plot_planning_time(self, save_path=None, plot_title = "Planning Time - TODO, TODO"):
        data, order = self._extract_data_for_metric("planning_time")
        self._plot_metric(data, order, "Planning Time [s]", plot_title, save_path)

    def plot_spline_fitting_time(self, save_path=None, plot_title = "Spline Fitting Time - TODO, TODO"):
        data, order = self._extract_data_for_metric("spline_fitting_time")
        self._plot_metric(data, order, "Spline Fitting Time [s]", plot_title, save_path)

    def plot_execution_time(self, save_path=None, plot_title = "Execution Time - TODO, TODO"):
        data, order = self._extract_data_for_metric("execution_time")
        self._plot_metric(data, order, "Execution Time [s]", plot_title, save_path)

    def plot_num_waypoints(self, save_path=None, plot_title = "Number of Waypoints - TODO, TODO"):
        data, order = self._extract_data_for_metric("number_of_waypoints_before_post_processing")
        self._plot_metric(data, order, "Number of Waypoints", plot_title, save_path)

def main():
    input_output_pairs = [
        (
            "/home/marcin/vm_shared/results/results_simulation_no_obstacle.npz",
            "/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/simulation_no_obstacle",
            "Simulation – No Obstacle"
        ),
        (
            "/home/marcin/vm_shared/results/results_simulation_obstacle.npz",
            "/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/simulation_obstacle",
            "Simulation – With Obstacle"
        ),
        (
            "/home/marcin/vm_shared/results/results_real_robot_no_obstacle.npz",
            "/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_no_obstacle",
            "Real Robot – No Obstacle"
        ),
        (
            "/home/marcin/vm_shared/results/results_real_robot_obstacle.npz",
            "/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_obstacle",
            "Real Robot – With Obstacle"
        ),
    ]

    for npz_path, base_dir, title_prefix in input_output_pairs:
        os.makedirs(base_dir, exist_ok=True)
        analysis = PlannerAnalysis(npz_path)

        analysis.plot_cartesian_path_length(
            save_path=os.path.join(base_dir, "cartesian_path_length.png"),
            plot_title=f"Cartesian Path Length – {title_prefix}"
        )
        analysis.plot_planning_time(
            save_path=os.path.join(base_dir, "planning_time.png"),
            plot_title=f"Planning Time – {title_prefix}"
        )
        analysis.plot_spline_fitting_time(
            save_path=os.path.join(base_dir, "spline_fitting_time.png"),
            plot_title=f"Spline Fitting Time – {title_prefix}"
        )
        analysis.plot_execution_time(
            save_path=os.path.join(base_dir, "execution_time.png"),
            plot_title=f"Execution Time – {title_prefix}"
        )
        analysis.plot_num_waypoints(
            save_path=os.path.join(base_dir, "num_waypoints.png"),
            plot_title=f"Number of Waypoints – {title_prefix}"
        )

if __name__ == "__main__":
    main()
