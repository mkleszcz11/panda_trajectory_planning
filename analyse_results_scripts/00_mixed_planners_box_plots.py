import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
import typing as t


class MixedPlannerBoxplot:
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

    def __init__(self, file_1: str, file_2: str):
        self.data_1 = np.load(file_1, allow_pickle=True)["results"]
        self.data_2 = np.load(file_2, allow_pickle=True)["results"]

    def get_group(self, display_name: str) -> str:
        for group, names in self.PLANNER_GROUP.items():
            if display_name in names:
                return group
        return "Other"

    def _extract(self, planner_name: str, metric: str, source: str):
        data_source = self.data_1 if source == "file_1" else self.data_2
        values = [
            float(entry[metric])
            for entry in data_source
            if entry.get("planning_successful", False)
            and self.DISPLAY_NAMES.get(entry.get("planner")) == planner_name
            and isinstance(entry.get(metric), (int, float))
        ]
        return values

    def plot_boxplot(
        self,
        planner_sources: t.List[t.Tuple[str, str]],
        metric: str,
        title: str,
        save_path: str,
        unit: str,
    ):
        data = []
        labels = []
        colors = []

        for planner, source in planner_sources:
            vals = self._extract(planner, metric, source)
            if vals:
                data.append(vals)
                labels.append(planner)
                group = self.get_group(planner)
                colors.append(self.GROUP_COLORS.get(group, "#888888"))

        # plt.rcParams.update({
        #     'font.size': 16,
        #     'axes.titlesize': 20,
        #     'axes.labelsize': 18,
        #     'xtick.labelsize': 16,
        #     'ytick.labelsize': 16,
        #     'legend.fontsize': 16,
        #     'legend.title_fontsize': 18,
        # })
        fig, ax = plt.subplots(figsize=(7, 6))
        bp = ax.boxplot(data, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        ax.set_xticklabels(labels, rotation=0, fontsize=18)
        ax.set_yticklabels(ax.get_yticks(), fontsize=18)
        ax.set_title(title, fontsize=20)
        if metric == "number_of_waypoints_before_post_processing":
            metric = "number_of_waypoints"
        ax.set_ylabel(metric.replace("_", " ").title()+ " " + unit, fontsize=18)
        ax.set_xlabel("Planner Variant", fontsize=18)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, pad_inches=0.1, dpi=300)
        plt.close()
        
        
plotter = MixedPlannerBoxplot(
    file_1="/home/marcin/vm_shared/results/results_simulation_obstacle_with_joint_distance.npz",
    file_2="/home/marcin/vm_shared/results/results_real_robot_obstacle_with_joint_distance.npz"
)
planner_sources=[
        ("PRM_2", "file_1"),
        ("PRM_5", "file_1"),
        ("PRM_13", "file_2")
    ]
metrics = ["planning_time", "cartesian_path_length", "execution_time", "spline_fitting_time", "number_of_waypoints_before_post_processing", "joint_distance"]
units = ["[s]", "[m]", "[s]", "[s]", "", "[rad]"]
titles = [
    "Planning Time Comparison Across\nSimulation (PRM_2, PRM_5) and\nReal Robot (PRM_13) - With Obstacle",
    "Cartesian Path Length Comparison Across\nSimulation (PRM_2, PRM_5) and\nReal Robot (PRM_13) - With Obstacle",
    "Execution Time Comparison Across\nSimulation (PRM_2, PRM_5) and\nReal Robot (PRM_13) - With Obstacle",
    "Spline Fitting Time Comparison Across\nSimulation (PRM_2, PRM_5) and\nReal Robot (PRM_13) - With Obstacle",
    "Number of Waypoints Comparison Across\nSimulation (PRM_2, PRM_5) and\nReal Robot (PRM_13) - With Obstacle",
    "Joint Distance Comparison Across\nSimulation (PRM_2, PRM_5) and\nReal Robot (PRM_13) - With Obstacle"
]
for metric, unit, title in zip(metrics, units, titles):
    plotter.plot_boxplot(
        planner_sources=planner_sources,
        metric=metric,
        unit=unit,
        title=title,
        save_path=f"/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_obstacle/custom_boxplot_{metric}.png"
    )
    
    
# plotter.plot_boxplot(
#     planner_sources=[
#         ("PRM_2", "file_1"),
#         ("PRM_5", "file_1"),
#         ("PRM_13", "file_2")
#     ],
#     metric="planning_time",
#     unit="[s]",
#     title="Planning Time Comparison Across Simulation (PRM_2, PRM_5)\nand Real Robot (PRM_13) - With Obstacle",
#     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_obstacle/custom_boxplot_planning_time.png"
# )
# plotter.plot_boxplot(
#     planner_sources=[
#         ("PRM_2", "file_1"),
#         ("PRM_5", "file_1"),
#         ("PRM_13", "file_2")
#     ],
#     metric="cartesian_path_length",
#     unit="[m]",
#     title="Cartesian Path Length Comparison Across Simulation (PRM_2, PRM_5)\nand Real Robot (PRM_13) - With Obstacle",
#     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_obstacle/custom_boxplot_cartesian_path_length.png"
# )
