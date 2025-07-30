import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from matplotlib import cm
import typing as t

class PlannerScatterPlotter:
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

    PLANNER_GROUP = {
        "RRT": ["RRT_1", "RRT_2", "RRT_3", "RRT_4", "RRT_5", "RRT_6"],
        "RRT*": ["RRTs_1", "RRTs_2", "RRTs_3", "RRTs_4", "RRTs_5"],
        "RRT-Connect": ["RRTc_1", "RRTc_2", "RRTc_3", "RRTc_4"],
        "PRM": [f"PRM_{i}" for i in range(1, 14)],
        "MoveIt": ["MoveIt"],
    }

    GROUP_COLORS = {
        "RRT": '#ffb000',
        "RRT*": '#785ef0',
        "RRT-Connect": '#dc267f',
        "PRM": '#fe6100',
        "MoveIt": '#648fff',
        # "Other": '#44aa99',
    }

    def __init__(self, npz_path: str):
        self.data = np.load(npz_path, allow_pickle=True)["results"]

    def get_group(self, name: str) -> str:
        for group, planners in self.PLANNER_GROUP.items():
            if name in planners:
                return group
        return "Other"

    def _safe_value(self, entry: dict, key: str):
        val = entry.get(key)
        if val is None or isinstance(val, str):
            return None
        try:
            val = float(val)
            if val < 0.01:
                return None
            return val
        except (ValueError, TypeError):
            return None

    def plot_parameters(
        self,
        x_param: str,
        y_param: str,
        x_unit: str = " ",
        y_unit: str = " ",
        save_path: str = None,
        title: str = None,
        add_averages: bool = False
    ):
        # Set global font sizes
        plt.rcParams.update({
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
        })

        x_vals, y_vals, colors, used_groups = [], [], [], set()
        group_values = {}

        for entry in self.data:
            if not entry.get("planning_successful", False):
                continue
            key = entry.get("planner")
            if key not in self.DISPLAY_NAMES:
                continue
            name = self.DISPLAY_NAMES[key]

            x = self._safe_value(entry, x_param)
            y = self._safe_value(entry, y_param)
            if x is None or y is None:
                continue

            group = self.get_group(name)
            x_vals.append(x)
            y_vals.append(y)
            colors.append(self.GROUP_COLORS[group])
            used_groups.add(group)

            if add_averages:
                group_values.setdefault(group, {"x": [], "y": []})
                group_values[group]["x"].append(x)
                group_values[group]["y"].append(y)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(x_vals, y_vals, c=colors, s=40, alpha=0.75)

        if add_averages:
            for group, values in group_values.items():
                if values["x"] and values["y"]:
                    mean_x = np.mean(values["x"])
                    mean_y = np.mean(values["y"])
                    ax.scatter(mean_x, mean_y, color=self.GROUP_COLORS[group], edgecolor='black',
                            s=130, marker='X', linewidths=1.5, zorder=5)

        ax.set_xlabel(x_param.replace("_", " ").capitalize() + " " + x_unit, fontsize=16)
        ax.set_ylabel(y_param.replace("_", " ").capitalize() + " " + y_unit, fontsize=16)
        ax.set_title(title or f"{y_param} vs. {x_param}".replace("_", " ").title(), fontsize=18)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=group,
                markerfacecolor=self.GROUP_COLORS[group], markersize=10)
            for group in self.GROUP_COLORS
            if group in used_groups
        ]
        if add_averages:
            legend_elements.append(
                Line2D([0], [0], marker='X', color='black', label="Group average",
                    markersize=10, linestyle='None')
            )

        ax.legend(
            handles=legend_elements,
            title="Planner Group",
            loc="upper right",
            frameon=True
        )

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_parameters_for_one_type(
        self,
        planner_type: str,
        x_param: str = "missing_parameter",
        x_param_unit: str = "",
        y_param: str = "missing_parameter",
        y_param_unit: str = "",
        save_path: str = None,
        title: str = None,
        planner_variants: t.Optional[t.List[str]] = None,
        add_averages: bool = False,
    ):
        assert planner_type in self.PLANNER_GROUP, f"Unknown planner type '{planner_type}'"

        all_planners = self.PLANNER_GROUP[planner_type]
        planners = (
            [p for p in all_planners if p in planner_variants]
            if planner_variants is not None else all_planners
        )

        x_vals, y_vals, colors, used_planners = [], [], [], set()
        cmap = plt.colormaps['tab10']
        planner_color_map = {planner: cmap(i) for i, planner in enumerate(planners)}
        averages = {planner: {'x': [], 'y': []} for planner in planners}

        for entry in self.data:
            if not entry.get("planning_successful", False):
                continue

            key = entry.get("planner")
            if key not in self.DISPLAY_NAMES:
                continue

            name = self.DISPLAY_NAMES[key]
            if name not in planners:
                continue

            x = self._safe_value(entry, x_param)
            y = self._safe_value(entry, y_param)
            if x is None or y is None:
                continue

            x_vals.append(x)
            y_vals.append(y)
            colors.append(planner_color_map[name])
            used_planners.add(name)

            if add_averages:
                averages[name]['x'].append(x)
                averages[name]['y'].append(y)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_vals, y_vals, c=colors, s=35, alpha=0.75)

        # Plot averages
        if add_averages:
            for planner, values in averages.items():
                if values['x'] and values['y']:
                    mean_x = np.mean(values['x'])
                    mean_y = np.mean(values['y'])
                    ax.scatter(mean_x, mean_y, color=planner_color_map[planner], edgecolor='black',
                            s=100, marker='X', linewidths=1.5, zorder=5)

        # Axes labels and title
        ax.set_xlabel(x_param.replace("_", " ").capitalize() + " " + x_param_unit)
        ax.set_ylabel(y_param.replace("_", " ").capitalize() + " " + y_param_unit)
        ax.set_title(title or f"{y_param} vs. {x_param} – {planner_type}".replace("_", " ").title())
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=planner,
                markerfacecolor=planner_color_map[planner], markersize=10)
            for planner in planners if planner in used_planners
        ]
        if add_averages:
            legend_elements.append(
                Line2D([0], [0], marker='X', color='black', label="Planner average",
                    markersize=10, linestyle='None')
            )

        ax.legend(
            handles=legend_elements,
            title="Planner",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            frameon=True
        )

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        else:
            plt.show()

    def plot_parameters_for_selected_variants(
        self,
        planner_variants: t.List[str],
        x_param: str = "missing_parameter",
        x_param_unit: str = "",
        y_param: str = "missing_parameter",
        y_param_unit: str = "",
        save_path: str = None,
        title: str = None,
        add_averages: bool = False,
    ):
        """
        Plot parameters for manually selected planner variants.

        Args:
            planner_variants: List of planner variant names (e.g., ["PRM_1", "RRTc_2"]).
            x_param: X-axis metric (e.g., "planning_time").
            y_param: Y-axis metric (e.g., "execution_time").
            save_path: Optional path to save the plot.
            title: Optional plot title.
            add_averages: Whether to mark average points.
        """
        x_vals, y_vals, colors, used_planners = [], [], [], set()
        cmap = plt.colormaps['tab10']
        planner_color_map = {planner: cmap(i) for i, planner in enumerate(planner_variants)}
        averages = {planner: {'x': [], 'y': []} for planner in planner_variants}

        for entry in self.data:
            if not entry.get("planning_successful", False):
                continue

            key = entry.get("planner")
            if key not in self.DISPLAY_NAMES:
                continue

            name = self.DISPLAY_NAMES[key]
            if name not in planner_variants:
                continue

            x = self._safe_value(entry, x_param)
            y = self._safe_value(entry, y_param)
            if x is None or y is None:
                continue

            x_vals.append(x)
            y_vals.append(y)
            colors.append(planner_color_map[name])
            used_planners.add(name)

            if add_averages:
                averages[name]['x'].append(x)
                averages[name]['y'].append(y)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_vals, y_vals, c=colors, s=35, alpha=0.75)

        if add_averages:
            for planner, values in averages.items():
                if values['x'] and values['y']:
                    mean_x = np.mean(values['x'])
                    mean_y = np.mean(values['y'])
                    ax.scatter(mean_x, mean_y, color=planner_color_map[planner], edgecolor='black',
                               s=100, marker='X', linewidths=1.5, zorder=5)

        ax.set_xlabel(x_param.replace("_", " ").capitalize() + x_param_unit)
        ax.set_ylabel(y_param.replace("_", " ").capitalize() + y_param_unit)
        ax.set_title(title or f"{y_param} vs. {x_param} – Selected Planners".replace("_", " ").title())
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=planner,
                   markerfacecolor=planner_color_map[planner], markersize=10)
            for planner in planner_variants if planner in used_planners
        ]
        if add_averages:
            legend_elements.append(
                Line2D([0], [0], marker='X', color='black', label="Planner average",
                       markersize=10, linestyle='None')
            )

        ax.legend(
            handles=legend_elements,
            title="Planner Variant",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            frameon=True
        )

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        else:
            plt.show()

# Main entry
if __name__ == "__main__":
    # plotter_sim_no_obstacle = PlannerScatterPlotter("/home/marcin/vm_shared/results/results_simulation_no_obstacle.npz")
    # # plotter_sim_no_obstacle.plot_parameters(
    # #     x_param="planning_time",
    # #     y_param="execution_time",
    # #     title="Execution Time vs. Planning Time - All Runs, Simulation, No Obstacle",
    # #     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/simulation_no_obstacle/00_execution_vs_planning_time.png"
    # # )
    # plotter_sim_no_obstacle.plot_parameters_for_one_type(
    #     planner_type="RRT",
    #     planner_variants=["RRT_1", "RRT_2", "RRT_3"],
    #     x_param="planning_time",
    #     y_param="execution_time",
    #     title="Execution Time vs. Planning Time – Simulation, No Obstacle (RRT_1 - RRT_3)",
    #     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/simulation_no_obstacle/00_execution_vs_planning_time_only_rrt_1-3.png",
    #     add_averages=True
    # )

    
    # plotter_sim_obstacle  = PlannerScatterPlotter("/home/marcin/vm_shared/results/results_simulation_obstacle.npz")
    # plotter_sim_obstacle.plot_parameters_for_one_type(
    #     planner_type="PRM",
    #     x_param="planning_time",
    #     y_param="execution_time",
    #     planner_variants=["PRM_1", "PRM_2", "PRM_3", "PRM_4", "PRM_5", "PRM_6"],
    #     title="Execution Time vs. Planning Time - All Runs, Simulation, With Obstacle",
    #     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/simulation_obstacle/00_execution_vs_planning_time_PRM.png"
    # )
    
    # plotter = PlannerScatterPlotter("/home/marcin/vm_shared/results/results_simulation_obstacle.npz")
    # plotter.plot_parameters_for_selected_variants(
    #     planner_variants=["PRM_3", "RRTc_4"],
    #     x_param="planning_time",
    #     y_param="execution_time",
    #     title="Execution Time vs. Planning Time – Mixed Planners",
    #     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/simulation_obstacle/00_execution_vs_planning_time_PRM3_RRTc4.png",
    #     add_averages=True
    # )
    # plotter_sim_obstacle.plot_parameters_for_one_type(
    #     planner_type="PRM",
    #     x_param="planning_time",
    #     y_param="execution_time",
    #     title="Execution Time vs. Planning Time – Simulation, With Obstacle (PRM)",
    #     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/simulation_obstacle/00_execution_vs_planning_time_only_prm.png"
    # )

    # plotter_sim_obstacle_with_joint_distance = PlannerScatterPlotter("/home/marcin/vm_shared/results/results_simulation_obstacle_with_joint_distance.npz")
    # # # plotter_sim_obstacle.plot_parameters(
    # # #     x_param="planning_time",
    # # #     y_param="execution_time",
    # # #     title="Execution Time vs. Planning Time - All Runs, Simulation, With Obstacle",
    # # #     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/simulation_obstacle/00_execution_vs_planning_time.png"
    # # # )
    # plotter_sim_obstacle_with_joint_distance.plot_parameters_for_one_type(
    #     planner_type="PRM",
    #     x_param="planning_time",
    #     y_param="joint_distance",
    #     title="Execution Time vs. Joint Distance – Simulation, With Obstacle (PRM)",
    #     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/simulation_obstacle/00_execution_vs_joint_distance_only_prm.png"
    # )
    # plotter_sim_real_robot_no_obstacle_with_joint_distance = PlannerScatterPlotter("/home/marcin/vm_shared/results/results_real_robot_no_obstacle_with_joint_distance.npz")
    # plotter_sim_real_robot_no_obstacle_with_joint_distance.plot_parameters_for_selected_variants(
    #     planner_variants=["MoveIt", "RRTc_1", "RRTs_1", "RRT_6", ],
    #     x_param="cartesian_path_length",
    #     x_param_unit="[m]",
    #     y_param="joint_distance",
    #     y_param_unit="[rad]",
    #     title="Cartesian Path Length vs. Joint Distance – Real Robot, No Obstacle (Selected Planners)",
    #     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_no_obstacle/00_cartesian_vs_joint_distance_selected_planners.png",
    #     add_averages=True
    # )
    plotter_sim_real_robot_obstacle_with_joint_distance = PlannerScatterPlotter("/home/marcin/vm_shared/results/results_real_robot_obstacle_with_joint_distance.npz")
    plotter_sim_real_robot_obstacle_with_joint_distance.plot_parameters(
        x_param="planning_time",
        y_param="execution_time",
        x_unit="[s]",
        y_unit="[s]",
        title="Execution Time vs. Planning Time - All Runs, Real Robot, With Obstacle",
        save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_obstacle/00_execution_vs_planning_time.png",
        add_averages=True
    )
    # # PLOT ONLY FOR RRTc
    # plotter_sim_real_robot_obstacle_with_joint_distance.plot_parameters_for_one_type(
    #     planner_type="RRT-Connect",
    #     x_param="cartesian_path_length",
    #     x_param_unit="[m]",
    #     y_param="joint_distance",
    #     y_param_unit="[rad]",
    #     title="Execution Time vs. Planning Time – Real Robot, With Obstacle (RRT-Connect)",
    #     save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_obstacle/00_cartesian_vs_joint_distance_only_rrtc.png",
    #     add_averages=True,
    #     planner_variants=["RRTc_1", "RRTc_2", "RRTc_3", "RRTc_4"]
    # )
    
    # EXECUTION VS PLANNING TIME FOR PRM_13 AND ALL RRTc
    plotter_sim_real_robot_obstacle_with_joint_distance.plot_parameters_for_selected_variants(
        planner_variants=["PRM_13", "RRTc_1", "RRTc_2", "RRTc_3", "RRTc_4"],
        x_param="planning_time",
        x_param_unit="[s]",
        y_param="execution_time",
        y_param_unit="[s]",
        title="Execution Time vs. Planning Time - Real Robot, With Obstacle (PRM_13 and RRT-Connect)",
        save_path="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_obstacle/00_execution_vs_planning_time_prm13_rrtc.png",
        add_averages=True
    )