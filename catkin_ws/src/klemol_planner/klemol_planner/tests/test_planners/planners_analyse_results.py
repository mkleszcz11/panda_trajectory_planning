import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

def analyze_results(filename='/tmp/planners_test_results.npz'):
    # Load the results
    data = np.load(filename, allow_pickle=True)
    results = data['results']
    all_tests = list(results)

    print(f"Loaded {len(all_tests)} test results from {filename}")

    # Collect statistics per planner
    summary = defaultdict(list)
    for entry in all_tests:
        planner = entry['planner']
        summary[planner].append(entry)

    # Print summary
    for planner, tests in summary.items():
        n = len(tests)
        avg_planning_time = np.mean([t['planning_time'] for t in tests if t['planning_time'] is not None])
        avg_execution_time = np.mean([t['execution_time'] for t in tests if t['execution_time'] is not None])
        avg_waypoints = np.mean([t['number_of_waypoints_before_post_processing'] for t in tests if t['number_of_waypoints_before_post_processing'] is not None])
        avg_cartesian_path = np.mean([t['cartesian_path_length'] for t in tests if t['cartesian_path_length'] is not None])

        print(f"\nPlanner: {planner}")
        print(f"  Number of tests: {n}")
        print(f"  Avg Planning Time: {avg_planning_time:.3f}s")
        print(f"  Avg Execution Time: {avg_execution_time:.3f}s")
        print(f"  Avg Waypoints: {avg_waypoints:.1f}")
        print(f"  Avg Cartesian Path Length: {avg_cartesian_path:.3f}")

    # Optional: Print details for each test
    print("\nDetailed Results:")
    for i, entry in enumerate(all_tests):
        print(f"\nTest {i+1} - Planner: {entry['planner']}")
        print(f"  Planning Time: {entry['planning_time']}")
        print(f"  Execution Time: {entry['execution_time']}")
        print(f"  Number of Steps: {entry['number_of_steps']}")
        print(f"  Number of Waypoints Before Post Processing: {entry['number_of_waypoints_before_post_processing']}")
        print(f"  Joint Travel Distances: {entry['joint_travel_distances']}")
        print(f"  Cartesian Path Length: {entry['cartesian_path_length']}")

    # Call the 3D plot function
    plot_3d_paths(all_tests)

def plot_3d_paths(all_tests):
    """
    Plots 3D Cartesian paths for each planner in one plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Color list for different planners

    planner_colors = {}
    color_idx = 0

    for entry in all_tests:
        planner = entry['planner']
        if planner not in planner_colors:
            planner_colors[planner] = colors[color_idx % len(colors)]
            color_idx += 1

        positions = np.array(entry['cartesain_positions'])
        if positions.shape[0] == 0:
            continue  # Skip if no data

        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        ax.plot(x, y, z, label=planner, color=planner_colors[planner])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D End-Effector Paths per Planner')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    analyze_results('/tmp/planner_test_results.npz')
