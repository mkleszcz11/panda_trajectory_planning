import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import typing as t
import matplotlib.gridspec as gridspec


class PlannerTrajectoryPlotter:
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
        "RRT": ["RRT_" + str(i) for i in range(1, 7)],
        "RRT*": ["RRTs_" + str(i) for i in range(1, 6)],
        "RRT-Connect": ["RRTc_" + str(i) for i in range(1, 5)],
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

    def get_group(self, planner_name: str) -> str:
        for group, names in self.PLANNER_GROUP.items():
            if planner_name in names:
                return group
        return "Other"

    def get_key_from_display_name(self, display_name: str) -> t.Optional[str]:
        for key, name in self.DISPLAY_NAMES.items():
            if name == display_name:
                return key
        return None

    def filter_runs(self, planner_names: t.List[str], run_indices: t.List[int]) -> t.List[dict]:
        selected = []
        seen = set()
        planner_keys = [self.get_key_from_display_name(name) for name in planner_names]

        for entry in self.data:
            planner = entry["planner"]
            loop_idx = entry["loop_index"]
            if planner in planner_keys and loop_idx in run_indices:
                if entry.get("cartesain_positions") is not None:
                    key = (planner, loop_idx)
                    if key not in seen:
                        selected.append(entry)
                        seen.add(key)
        return selected

    def plot_views_for_thesis(
        self,
        planner_names: t.List[str],
        run_indices: t.List[int],
        color_per_run: bool = False,
        color_per_group: bool = False,
        output_dir: str = "plots",
        plot_title: str = "TITLE",
        file_name: str = "trajectory_view.png",
        fixed_xlim: t.Tuple[float, float] = (0, 1),
        fixed_ylim: t.Tuple[float, float] = (-0.5, 0.5),
        fixed_zlim: t.Tuple[float, float] = (0.05, 1.05)
    ):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
        import os

        os.makedirs(output_dir, exist_ok=True)
        selected_runs = self.filter_runs(planner_names, run_indices)

        # Prepare variant-level color mapping if needed
        if color_per_group:
            unique_variants = sorted(set(self.DISPLAY_NAMES.get(e["planner"], e["planner"]) for e in selected_runs))
            variant_to_color = {name: plt.colormaps["tab10"](i % 10) for i, name in enumerate(unique_variants)}

        # Setup figure and axes
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.3)
        ax_3d = fig.add_subplot(gs[0], projection='3d')
        ax_xy = fig.add_subplot(gs[1])
        ax_xz = fig.add_subplot(gs[2])
        fig.suptitle(plot_title, fontsize=16, y=0.87)

        seen_labels = set()
        all_x, all_y, all_z = [], [], []

        for i, entry in enumerate(selected_runs):
            planner_key = entry["planner"]
            planner_display_name = self.DISPLAY_NAMES.get(planner_key, planner_key)
            group = self.get_group(planner_display_name)

            positions = np.array(entry["cartesain_positions"])
            if positions.shape[0] == 0:
                continue

            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)

            # Determine color and label
            if color_per_run:
                color = plt.colormaps["tab10"](i % 10)
                label = f"{planner_display_name} Run {entry['loop_index']}"
            elif color_per_group:
                color = variant_to_color[planner_display_name]
                label = planner_display_name
            else:
                color = self.GROUP_COLORS.get(group, "#aaaaaa")
                label = group

            # Prevent label duplication in legend
            legend_label = label if label not in seen_labels else None
            seen_labels.add(label)

            # Plot
            ax_3d.plot(x, y, z, label=legend_label, color=color)
            ax_xy.plot(y, x, color=color)
            ax_xz.plot(x, z, color=color)

        # Set consistent axis ranges
        if all_x and all_y and all_z:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            min_z, max_z = min(all_z), max(all_z)

            center_x = 0.5 * (min_x + max_x)
            center_y = 0.5 * (min_y + max_y)
            center_z = 0.5 * (min_z + max_z)

            range_x = max_x - min_x
            range_y = max_y - min_y
            range_z = max_z - min_z
            half_range = max(range_x, range_y, range_z) / 2

            fixed_xlim = (center_x - half_range, center_x + half_range)
            fixed_ylim = (center_y - half_range, center_y + half_range)
            fixed_zlim = (center_z - half_range, center_z + half_range)

        # 3D Axes
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_xlim(*fixed_xlim)
        ax_3d.set_ylim(*fixed_ylim)
        ax_3d.set_zlim(*fixed_zlim)
        ax_3d.legend(
            loc='center left',
            bbox_to_anchor=(-0.4, 0.5),
            frameon=True,
            fontsize=12,
            title="Planner",
            title_fontsize=13
        )

        # 2D XY
        ax_xy.set_xlabel('Y (m)')
        ax_xy.set_ylabel('X (m)')
        ax_xy.set_xlim(*fixed_ylim)
        ax_xy.set_ylim(*fixed_xlim)
        ax_xy.set_aspect('equal')
        ax_xy.grid(True)

        # 2D XZ
        ax_xz.set_xlabel('X (m)')
        ax_xz.set_ylabel('Z (m)')
        ax_xz.set_xlim(*fixed_xlim)
        ax_xz.set_ylim(*fixed_zlim)
        ax_xz.set_aspect('equal')
        ax_xz.grid(True)

        # Save
        save_path = os.path.join(output_dir, file_name)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

def main():
    plotter = PlannerTrajectoryPlotter("/home/marcin/vm_shared/results/results_real_robot_obstacle.npz")
    plotter.plot_views_for_thesis(
        planner_names=["MoveIt", "RRTc_3", "PRM_13"],
        run_indices=[8],
        color_per_run=False,
        color_per_group=True,
        output_dir="/home/marcin/DTU/000_thesis/panda_trajectory_planning/analyse_results_scripts/plots/real_robot_obstacle",
        plot_title="Trajectories for one run - MoveIt, RRTc_3, PRM_13 (Real Robot, With Obstacle)",
        file_name="moveit_rrtc3_prm13_one_run_paths_obstacle.png"
    )

if __name__ == "__main__":
    main()