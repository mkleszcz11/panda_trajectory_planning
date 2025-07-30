import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# FOR SIMULATION RANDOM MOVEMENT - START #
acronym_order = [
    "rrt-step_02-bias_02",
    "rrt-step_04-bias_02",
    "rrt-step_06-bias_02",
    "rrt-step_02-bias_04",
    "rrt-step_04-bias_04",
    "rrt-step_06-bias_04",
    # "rrt_star-step_02-bias_02-rewire_061",
    "rrt_star-step_04-bias_02-rewire_081",
    "rrt_star-step_06-bias_02-rewire_121",
    "rrt_star-step_04-bias_04-rewire_081",
    "rrt_star-step_06-bias_04-rewire_121",
    "rrt_with_connecting-step_01-bias_01",
    "rrt_with_connecting-step_02-bias_02",
    "rrt_with_connecting-step_04-bias_02",
    "rrt_with_connecting-step_06-bias_06",
    "prm_1000_samples_not_restricted",
    "prm_10000_samples_not_restricted",
    "prm_100000_samples_not_restricted",
    "prm_1000_samples_not_restricted_diff_w",
    "prm_10000_samples_not_restricted_diff_w",
    "prm_100000_samples_not_restricted_diff_w",
]

# Assign acronyms like 'RRT_1', 'RRTs_2', etc.
planner_acronyms = {
    "rrt-step_02-bias_02": "RRT_1",
    "rrt-step_04-bias_02": "RRT_2",
    "rrt-step_06-bias_02": "RRT_3",
    "rrt-step_02-bias_04": "RRT_4",
    "rrt-step_04-bias_04": "RRT_5",
    "rrt-step_06-bias_04": "RRT_6",
    # "rrt_star-step_02-bias_02-rewire_061": "RRTs_1",
    "rrt_star-step_04-bias_02-rewire_081": "RRTs_2",
    "rrt_star-step_06-bias_02-rewire_121": "RRTs_3",
    "rrt_star-step_04-bias_04-rewire_081": "RRTs_4",
    "rrt_star-step_06-bias_04-rewire_121": "RRTs_5",
    "rrt_with_connecting-step_01-bias_01": "RRTc_1",
    "rrt_with_connecting-step_02-bias_02": "RRTc_2",
    "rrt_with_connecting-step_04-bias_02": "RRTc_3",
    "rrt_with_connecting-step_06-bias_06": "RRTc_4",
    "prm_1000_samples_not_restricted": "PRM_1",
    "prm_10000_samples_not_restricted": "PRM_2",
    "prm_100000_samples_not_restricted": "PRM_3",
    "prm_1000_samples_not_restricted_diff_w": "PRM_4",
    "prm_10000_samples_not_restricted_diff_w": "PRM_5",
    "prm_100000_samples_not_restricted_diff_w": "PRM_6",
}
# FOR SIMULATION RANDOM MOVEMENT - STOP #

# Load test data
filename = r'/home/marcin/vm_shared/results/results_simulation_obstacle.npz'
# Output directory
output_dir = "plots_simulation_obstacle"


# END OF FILE RELATED STUFF #
data = np.load(filename, allow_pickle=True)
results = list(data['results']) 

os.makedirs(output_dir, exist_ok=True)

# Group entries by planner name
summary = defaultdict(list)
for entry in results:
    summary[entry['planner']].append(entry)

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

planner_groups = {
    planner_acronyms[name]: get_group(name) for name in acronym_order
}
sorted_acronyms = [planner_acronyms[name] for name in acronym_order]

group_colors = {
    "RRT": '#ffb000',
    "RRT*": '#785ef0',
    "RRT-Connect": '#dc267f',
    "PRM": '#fe6100',
    "Other": '#44aa99'
}

# Style
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# Metrics to visualize
metrics = {
    "Planning Time [s]": lambda e: e['planning_time'],
    "Spline Fitting Time [s]": lambda e: e['spline_fitting_time'],
    "Execution Time [s]": lambda e: e['execution_time'],
    "Waypoints": lambda e: e['number_of_waypoints_before_post_processing'],
    "Cartesian Path Length [m]": lambda e: e['cartesian_path_length'],
}

# Plot each metric
for metric_name, extractor in metrics.items():
    labels, values, colors = [], [], []

    for acronym in sorted_acronyms:
        planner = [k for k, v in planner_acronyms.items() if v == acronym][0]
        entries = summary[planner]
        valid = [extractor(e) for e in entries if e.get("planning_successful", True) and extractor(e) is not None]
        if valid:
            labels.append(acronym)
            values.append(np.mean(valid))
            colors.append(group_colors[planner_groups[acronym]])

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = range(len(labels))
    ax.bar(x_pos, values, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    ax.set_ylabel(metric_name, fontsize=16)
    ax.set_title(f"Average {metric_name}", fontsize=18)
    ax.tick_params(axis='y', labelsize=14)

    # Legend with larger font
    handles = [plt.Rectangle((0,0),1,1,color=group_colors[g]) for g in group_colors]
    if metric_name == "Cartesian Path Length [m]" or metric_name == "Average Execution Time [s]":
        ax.legend(
            handles, 
            group_colors.keys(), 
            title="Planner Type", 
            fontsize=14, 
            title_fontsize=15,
            loc='lower right'
        )
    else:
        ax.legend(
            handles, 
            group_colors.keys(), 
            title="Planner Type", 
            fontsize=14, 
            title_fontsize=15
        )

    fig.tight_layout()
    fname = metric_name.lower().replace(' ', '_').replace('[', '').replace(']', '').replace('/', '') + '.png'
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)

# Plot Success/Failure counts with colors
successes, failures, sf_colors = [], [], []

for acronym in sorted_acronyms:
    planner = [k for k, v in planner_acronyms.items() if v == acronym][0]
    entries = summary[planner]
    s = sum(1 for e in entries if e.get("planning_successful", True))
    f = sum(1 for e in entries if not e.get("planning_successful", True))
    successes.append(s)
    failures.append(f)
    sf_colors.append(group_colors[planner_groups[acronym]])

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = range(len(sorted_acronyms))

ax.bar(x_pos, successes, label='S', color=sf_colors)
ax.bar(x_pos, failures, bottom=successes, label='F', color='gray')

ax.set_xticks(x_pos)
ax.set_xticklabels(sorted_acronyms, rotation=45, ha='right', fontsize=14)
ax.set_ylabel("Count", fontsize=16)
ax.set_title("Success (S) and Failure (F) Counts per Planner (Ordered)", fontsize=18)
ax.tick_params(axis='y', labelsize=14)
ax.legend(fontsize=14, title_fontsize=15)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
fig.tight_layout()
fname = "success_failure_counts.png"
fig.savefig(os.path.join(output_dir, fname), dpi=300)
plt.close(fig)

plot_files = os.listdir(output_dir)
df = pd.DataFrame({"Plot Name": plot_files})
