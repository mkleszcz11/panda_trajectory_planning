import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load test data
filename = '/home/marcin/vm_shared/results/planner_random_30_loops_test_results.npz'
data = np.load(filename, allow_pickle=True)
results = list(data['results']) 

# Output directory
output_dir = "plots_random_movement"
os.makedirs(output_dir, exist_ok=True)

# Group entries by planner name
summary = defaultdict(list)
for entry in results:
    summary[entry['planner']].append(entry)

acronym_order = [
    "rrt-step_02-bias_02",
    "rrt-step_04-bias_02",
    "rrt-step_06-bias_02",
    "rrt-step_02-bias_04",
    "rrt-step_04-bias_04",
    "rrt-step_06-bias_04",
    "rrt_star-step_02-bias_02-rewire_061",
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

# Acronym assignment
planner_acronyms = {name: chr(65 + i) for i, name in enumerate(acronym_order)}

# Planner group assignment
def get_group(name):
    if "rrt_star" in name:
        return "RRT*"
    elif "rrt_with_connecting" in name:
        return "RRT-Connect"
    elif "rrt" in name:
        return "RRT"
    elif "prm" in name:
        return "PRM"
    else:
        return "Other"

planner_groups = {planner_acronyms[name]: get_group(name) for name in acronym_order}
sorted_acronyms = [planner_acronyms[name] for name in acronym_order]
group_colors = {
    "PRM": '#1f77b4',        # blue
    "RRT": '#ff7f0e',        # orange
    "RRT*": '#2ca02c',       # green
    "RRT-Connect": '#d62728' # red
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
    ax.bar(labels, values, color=colors)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Average {metric_name}")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Create a color legend by group
    handles = [plt.Rectangle((0,0),1,1,color=group_colors[g]) for g in group_colors]
    ax.legend(handles, group_colors.keys(), title="Planner Type")

    fig.tight_layout()
    fname = "random_movements_" + metric_name.lower().replace(' ', '_').replace('[', '').replace(']', '').replace('/', '') + '.png'
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
ax.bar(sorted_acronyms, successes, label='S', color=sf_colors)
ax.bar(sorted_acronyms, failures, bottom=successes, label='F', color='gray')
ax.set_ylabel("Count")
ax.set_title("Success (S) and Failure (F) Counts per Planner (Ordered)")
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "random_movements_success_failure_counts.png"), dpi=300)
plt.close(fig)

import pandas as pd
plot_files = os.listdir(output_dir)
df = pd.DataFrame({"Plot Name": plot_files})
