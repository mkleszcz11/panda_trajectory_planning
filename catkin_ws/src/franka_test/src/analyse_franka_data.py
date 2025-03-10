#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the movement log file
filename = "/tmp/franka_movement_log.csv"

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: The file {filename} was not found. Make sure the motion script has run.")
    exit()

# Separate data by method
trac_ik_data = df[df["method"] == "TRAC-IK"]
planner_data = df[df["method"] == "Trajectory Planner"]

# Compare Execution Times
plt.figure(figsize=(8, 5))
methods = ["TRAC-IK", "Trajectory Planner"]
execution_times = [trac_ik_data["execution_time"].sum(), planner_data["execution_time"].sum()]

plt.bar(methods, execution_times, color=['blue', 'orange'])
plt.xlabel("Control Method")
plt.ylabel("Total Execution Time (s)")
plt.title("Execution Time Comparison")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Compare Joint Positions (First Movement)
if not trac_ik_data.empty and not planner_data.empty:
    # Extract the first recorded movement (excluding the initial joint move)
    trac_ik_joints = eval(trac_ik_data.iloc[1]["joint_positions"])  # Convert string to list
    planner_joints = eval(planner_data.iloc[1]["joint_positions"])  # Convert string to list

    joint_indices = np.arange(len(trac_ik_joints))

    plt.figure(figsize=(8, 5))
    plt.plot(joint_indices, trac_ik_joints, 'bo-', label="TRAC-IK")
    plt.plot(joint_indices, planner_joints, 'ro-', label="Trajectory Planner")
    plt.xlabel("Joint Index")
    plt.ylabel("Joint Angle (radians)")
    plt.title("Joint Positions for First Movement")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Not enough data to compare joint positions.")

print("Analysis Complete!")
