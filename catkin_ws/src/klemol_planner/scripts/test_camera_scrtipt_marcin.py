
from klemol_planner.camera_utils.camera_operations import CameraOperations
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.goals.point_with_orientation import PointWithOrientation


if __name__ == "__main__":
        camera_operations = CameraOperations()
        panda_transformations = PandaTransformations(cam_operations=camera_operations)

       # Get all marker transforms in camera frame
        marker_transforms = camera_operations.get_marker_transforms()

        # Prepare a dictionary for visualization
        visualisation_frames = {}

        # Iterate over all detected corners
        for corner_name in ["corner_0", "corner_1", "corner_2", "corner_3"]:
            if corner_name not in marker_transforms:
                print(f"[WARN] {corner_name} not detected.")
                continue

            # Extract translation
            x, y, z = marker_transforms[corner_name][:3, 3]

            # Construct a point in the camera frame
            corner_cam = PointWithOrientation(x, y, z, 0.0, 0.0, 0.0)

            # Transform to base frame
            corner_base = panda_transformations.transform_point(corner_cam, 'camera', 'base')

            # Store for visualization
            visualisation_frames[f"{corner_name}_in_camera_frame"] = corner_base.as_matrix()

        # Detect object and add to visualisation frames
        

        # Visualise
        panda_transformations.visusalise_environment(visualisation_frames, show_points_orientation=False)
