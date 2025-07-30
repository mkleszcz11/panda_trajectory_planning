import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict


class PlannerTimingBoxplotter:
    def __init__(self, npz_path: str, display_names: Dict[str, str]):
        self.data = np.load(npz_path, allow_pickle=True)["results"]
        self.display_names = display_names

    def get_valid_timings(self, planner_key: str):
        planning, spline, execution = [], [], []
        for entry in self.data:
            if not entry.get("planning_successful", False):
                continue
            if entry.get("planner") != planner_key:
                continue
            p, s, e = entry.get("planning_time"), entry.get("spline_fitting_time"), entry.get("execution_time")
            if isinstance(p, (float, int)) and p > 0.01:
                planning.append(p)
            if isinstance(s, (float, int)) and s > 0.001:
                spline.append(s)
            if isinstance(e, (float, int)) and e > 0.01:
                execution.append(e)
        return planning, spline, execution

    def plot_boxplots(self, planner_variants: List[str], save_path: Optional[str] = None, title: Optional[str] = None):
        data = []
        xtick_positions = []
        xtick_labels = []
        colors = []
        position = 1

        for i, planner in enumerate(planner_variants):
            planner_key = [k for k, v in self.display_names.items() if v == planner]
            if not planner_key:
                continue
            planner_key = planner_key[0]

            planning, spline, execution = self.get_valid_timings(planner_key)
            data.extend([planning, spline, execution])
            colors.extend(['#648fff', '#ffb000', '#dc267f'])

            mid_position = position + 1
            xtick_positions.append(mid_position)
            xtick_labels.append(planner)
            position += 4  # leave 1 unit gap between groups

        fig, ax = plt.subplots(figsize=(1.2 * len(planner_variants) * 3, 7))

        box_positions = [i * 4 + j + 1 for i in range(len(planner_variants)) for j in range(3)]
        bplot = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.6,
            positions=box_positions,
            showfliers=False  # Remove outliers
        )

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        # Vertical separators between planners
        for i in range(1, len(planner_variants)):
            ax.axvline(i * 4 - 0.0, linestyle='--', color='gray', linewidth=1)

        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=15)
        ax.set_xlabel("Planner Variant", fontsize=17)
        ax.set_ylabel("Time [s]", fontsize=17)
        ax.set_title(title or "Planning, Spline, Execution Times", fontsize=20)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        legend_labels = ['Planning', 'Spline', 'Execution']
        legend_colors = ['#648fff', '#ffb000', '#dc267f']
        handles = [plt.Line2D([0], [0], color=c, lw=10) for c in legend_colors]
        ax.legend(handles, legend_labels, loc="upper right", fontsize=14)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        else:
            plt.show()

DISPLAY_NAMES = {
    "rrt_with_connecting-step_02-bias_02": "RRTc_1",
    "rrt_with_connecting-step_04-bias_02": "RRTc_2",
    "rrt_with_connecting-step_01-bias_01": "RRTc_3",
    "rrt_with_connecting-step_06-bias_06": "RRTc_4",
    "prm_10000_samples_restricted_fixed_tool": "PRM_13",
}

plotter = PlannerTimingBoxplotter("/home/marcin/vm_shared/results/results_real_robot_obstacle_with_joint_distance.npz", DISPLAY_NAMES)
plotter.plot_boxplots(
    planner_variants=["RRTc_1", "RRTc_2", "RRTc_3", "RRTc_4", "PRM_13"],
    save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_obstacle/boxplot_times.png",
    title="Execution, Planning, and Spline Fitting Times (Real Robot, Obstacle)"
)