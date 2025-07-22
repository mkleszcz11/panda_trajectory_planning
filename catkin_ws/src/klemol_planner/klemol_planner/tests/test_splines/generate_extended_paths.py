import numpy as np
import os
import typing as t

OUTPUT_DIR = "/home/neurorobotic_student/panda_trajectory_planning/catkin_ws/src/klemol_planner/klemol_planner/tests/test_splines/extended_paths"  # make sure this exists or will be created
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_extended_path(
    path: t.List[np.ndarray], alpha: int, perturbation_range: float = 0.03
) -> t.List[np.ndarray]:
    """
    Generate a densified path with perturbed intermediate points,
    preserving original waypoints exactly once.
    """
    if alpha < 1:
        raise ValueError("Alpha must be ≥ 1")

    extended_path = []

    for i in range(len(path) - 1):
        q_start = path[i]
        q_end = path[i + 1]

        if i == 0:
            extended_path.append(q_start)  # Add the start only once

        for j in range(1, alpha):
            t = j / alpha
            q_interp = (1 - t) * q_start + t * q_end
            delta_q = np.random.uniform(-perturbation_range, perturbation_range, size=7)
            q_new = q_interp + delta_q
            extended_path.append(q_new)

        extended_path.append(q_end)  # Always include original endpoint

    print(f"Generated extended path: {extended_path} for (alpha={alpha})")
    return extended_path

def main():
    main_path = [
        np.array([0.0, -0.786, 0.0, -2.356, 0.0, 1.572, 0.785]),
        np.array([0.682, 0.025, -0.038, -2.118, -0.002, 2.145, 1.507]),
        np.array([0.579, 0.303, 0.062, -2.237, -0.038, 2.542, 1.533]),
        np.array([-0.456, 0.084, 0.498, -1.988, -0.042, 2.065, 0.858]),
        np.array([-0.542, -0.626, 0.055, -2.502, 0.831, 2.165, 0.816]),
        np.array([-0.261, -0.068, -0.534, -2.334, -0.117, 2.260, -0.060]),
    ]

    for alpha in [1, 2, 3, 5, 7, 10]:
        extended = generate_extended_path(main_path, alpha)
        filename = f"extended_path_alpha_{alpha}.npy"
        # np.save(os.path.join(OUTPUT_DIR, filename), np.array(extended))
        print(f"Saved: {filename} with {len(extended)} waypoints.")

if __name__ == "__main__":
    main()