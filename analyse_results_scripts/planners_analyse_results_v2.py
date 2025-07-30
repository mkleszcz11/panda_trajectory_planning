import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

def analyze_results(filename='/tmp/planner_test_results.npz', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(filename, allow_pickle=True)
    results = list(data['results'])
    grouped_by_loop = defaultdict(list)
    
    print(f"Loaded {len(results)} test results from {filename}")

    for entry in results:
        loop_idx = entry.get('loop_index', -1)
        grouped_by_loop[loop_idx].append(entry)

    for loop_idx, loop_entries in grouped_by_loop.items():
        fig = plt.figure(figsize=(14, 12), dpi=300)
        axes = [
            fig.add_subplot(2, 2, 1, projection='3d'),
            fig.add_subplot(2, 2, 2, projection='3d'),
            fig.add_subplot(2, 2, 3, projection='3d'),
            fig.add_subplot(2, 2, 4, projection='3d'),
        ]
        view_angles = [
            (30, 45), (90, 90), (0, 90), (0, 0)
        ]
        planner_colors = {}
        color_idx = 0
        cmap = plt.get_cmap("tab10")

        for entry in loop_entries:
            planner = entry['planner']
            pos = np.array(entry['cartesain_positions'])
            if pos.shape[0] == 0:
                continue

            if planner not in planner_colors:
                planner_colors[planner] = cmap(color_idx % 10)
                color_idx += 1

            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            for ax, (elev, azim) in zip(axes, view_angles):
                ax.plot(x, y, z, label=planner, color=planner_colors[planner], linewidth=1.5)

        for idx, ax in enumerate(axes):
            ax.view_init(elev=view_angles[idx][0], azim=view_angles[idx][1])
            ax.set_title(f"View {idx+1}", fontsize=12)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")

        fig.suptitle(f"Loop {loop_idx} - Cartesian Paths", fontsize=16)

        # Add legend below all subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        plot_path = os.path.join(output_dir, f"loop_{loop_idx}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved high-resolution plot for loop {loop_idx} to {plot_path}")

analyze_results(r'C:\Users\ander\OneDrive - Danmarks Tekniske Universitet\Master Thesis\AM MK Thesis Panda Robot\vm_shared\results\planner_fixed_no_obstacle_30_loops_test_results.npz')
