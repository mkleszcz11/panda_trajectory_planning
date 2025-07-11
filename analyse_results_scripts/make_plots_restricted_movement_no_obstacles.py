import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Config ---
filename = '/home/marcin/vm_shared/results/planner_fixed_no_obstacle_30_loops_test_results.npz'
output_dir = "plots_fixed_no_obstacle"
os.makedirs(output_dir, exist_ok=True)

# --- Load data ---
data = np.load(filename, allow_pickle=True)
results = list(data['results'])

# --- Group entries by planner name ---
summary = defaultdict(list)
for entry in results:
    summary[entry['planner']].append(entry)

# --- Define planners in desired display order ---
planner_order = [
    # RRT
    "rrt-step_02-bias_02",
    "rrt-step_04-bias_02",
    "rrt-step_06-bias_02",
    "rrt-step_02-bias_04",
    "rrt-step_04-bias_04",
    "rrt-step_06-bias_04",

    # RRT*
    "rrt_star-step_02-bias_02-rewire_061",
    "rrt_star-step_04-bias_02-rewire_081",
    "rrt_star-step_06-bias_02-rewire_121",
    "rrt_star-step_04-bias_04-rewire_081",
    "rrt_star-step_06-bias_04-rewire_121",

    # RRT-Connect
    "rrt_with_connecting-step_01-bias_01",
    "rrt_with_connecting-step_02-bias_02",
    "rrt_with_connecting-step_04-bias_02",
    "rrt_with_connecting-step_06-bias_06",

    # PRM - full workspace
    "prm_1000_samples_not_restricted",
    "prm_1000_samples_not_restricted_diff_w",
    "prm_10000_samples_not_restricted",
    "prm_10000_samples_not_restricted_diff_w",
    "prm_100000_samples_not_restricted",
    "prm_100000_samples_not_restricted_diff_w",

    # PRM - noc (no collision)
    "prm_1000_samples_not_restricted_noc",
    "prm_10000_samples_not_restricted_noc",
    "prm_100000_samples_not_restricted_noc",

    # PRM - restricted workspace + noc + diff weights
    "prm_1000_diff_w_samples_restricted_noc",
    "prm_10000_diff_w_samples_restricted_noc",
    "prm_100000_diff_w_samples_restricted_noc"
]

# --- Acronym Mapping ---
acronyms = {name: chr(65 + i) for i, name in enumerate(planner_order)}

# --- Group Colors ---
def get_group(name):
    if "rrt_star" in name:
        return "RRT*"
    elif "rrt_with_connecting" in name:
        return "RRT-Connect"
    elif "rrt" in name:
        return "RRT"
    elif "prm" in name:
        return "PRM"
    return "Other"

group_colors = {
    "PRM": '#1f77b4',
    "RRT": '#ff7f0e',
    "RRT*": '#2ca02c',
    "RRT-Connect": '#d62728',
    "Other": '#888888'
}

planner_groups = {acronyms[p]: get_group(p) for p in planner_order}
sorted_acronyms = [acronyms[p] for p in planner_order]

# --- Metrics ---
metrics = {
    "Planning Time [s]": lambda e: e['planning_time'],
    "Spline Fitting Time [s]": lambda e: e['spline_fitting_time'],
    "Execution Time [s]": lambda e: e['execution_time'],
    "Waypoints": lambda e: e['number_of_waypoints_before_post_processing'],
    "Cartesian Path Length [m]": lambda e: e['cartesian_path_length'],
}

# --- Style ---
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# --- Plot Boxplots ---
for metric_name, extractor in metrics.items():
    data_per_planner, box_colors, labels = [], [], []

    for planner in planner_order:
        entries = summary[planner]
        valid = [extractor(e) for e in entries if e.get("planning_successful", True) and extractor(e) is not None]
        if valid:
            data_per_planner.append(valid)
            acronym = acronyms[planner]
            labels.append(acronym)
            box_colors.append(group_colors[planner_groups[acronym]])

    fig, ax = plt.subplots(figsize=(12, 6))
    box = ax.boxplot(data_per_planner, patch_artist=True, labels=labels, showfliers=True)

    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)

    ax.set_title(f"{metric_name} (Fixed Start → Goal, No Obstacle)")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Planner Acronym (A–U)")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Group legend
    handles = [plt.Rectangle((0,0),1,1,color=c) for c in group_colors.values()]
    ax.legend(handles, group_colors.keys(), title="Planner Type")

    fname = "fixed_movement_" + metric_name.lower().replace(' ', '_').replace('[', '').replace(']', '').replace('/', '') + '.png'
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)
