import numpy as np
import os

# Configure
LOAD_DIR = "/home/marcin/vm_shared/000_real_robot/planners"
BASE_DIR = "/home/marcin/vm_shared/results"
NO_OBS_FILES = sorted([
    "real_robot_planner_no_obstace_5_loops_v1.npz",
    "real_robot_planner_no_obstace_5_loops_v2.npz",
    "real_robot_planner_no_obstace_5_loops_v3.npz",
    "real_robot_planner_no_obstace_5_loops_v4.npz",
    "moveit_no_obstacle_results.npz",
])
OBS_FILES = sorted([
    "real_robot_planner_obstace_5_loops_v4.npz",
    "real_robot_planner_obstace_5_loops_v5.npz",
    "moveit_obstacle_results.npz",
])
PRM_SIM_FILES = sorted([
    "planner_obstacle_30_loops_test_results.npz",
    "planner_prm_obstacle_30_loops_test_results.npz",
])

SAVE_NO_OBS = os.path.join(BASE_DIR, "results_real_robot_no_obstacle.npz")
SAVE_OBS = os.path.join(BASE_DIR, "results_real_robot_obstacle.npz")
SAVE_PRM_SIM = os.path.join(BASE_DIR, "results_simulation_obstacle.npz")

def merge_npz_files(file_list, output_path):
    merged = {}
    for file in file_list:
        data = np.load(os.path.join(LOAD_DIR, file), allow_pickle=True)
        for key in data:
            if key not in merged:
                merged[key] = list(data[key])
            else:
                merged[key].extend(data[key])
    # Save as numpy arrays
    np.savez(output_path, **{k: np.array(v, dtype=object) for k, v in merged.items()})

# Merge and save
merge_npz_files(NO_OBS_FILES, SAVE_NO_OBS)
merge_npz_files(OBS_FILES, SAVE_OBS)
# merge_npz_files(PRM_SIM_FILES, SAVE_PRM_SIM)

print("✅ Merging complete.")