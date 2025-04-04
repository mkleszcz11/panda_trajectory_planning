import typing as t
import pyrealsense2 as rs
import numpy as np
import cv2

class CameraOperations:
    def __init__(self):
        """
        Initialize the RealSense D455 camera and align depth to color stream.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        self.marker_length = 0.04  # [m] physical marker size

        # Load camera intrinsics from RealSense
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(intr.coeffs)

    def get_image(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Capture one color + depth frame.

        Returns:
            color_image: BGR image
            depth_frame: aligned depth frame
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Could not get frames from RealSense camera")

        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame

    def find_aruco_codes_in_the_image(self) -> t.List[t.Tuple[int, np.ndarray]]:
        """
        Detect ArUco markers and return translation vectors.

        Returns:
            List of (marker_id, translation_vector) in meters.
        """
        color_image, _ = self.get_image()
        corners, ids, _ = self.detector.detectMarkers(color_image)
        if ids is None:
            return []

        detected_markers = []

        for i, corner in enumerate(corners):
            rvec, tvec = self.estimate_pose_single_marker(
                corner, self.marker_length, self.camera_matrix, self.dist_coeffs)

            # Approximate center of the marker
            cx = int(corner[0][:, 0].mean())
            cy = int(corner[0][:, 1].mean())

            # Get aligned depth frame again to extract accurate Z
            _, depth_frame = self.get_image()
            z_depth = depth_frame.get_distance(cx, cy)
            rvec, tvec = self.estimate_pose_single_marker(
                corner, self.marker_length, self.camera_matrix, self.dist_coeffs, z_override=z_depth
            )

            # Replace only the Z component with sensor-based depth
            tvec[2][0] = z_depth

            cv2.aruco.drawDetectedMarkers(color_image, corners)
            # Project marker center (0,0,0) into image to verify alignment
            image_point, _ = cv2.projectPoints(
                np.array([[0.0, 0.0, 0.0]]),  # marker origin
                rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            u, v = tuple(image_point[0][0].astype(int))
            cv2.circle(color_image, (u, v), 5, (255, 0, 0), -1)  # Yellow dot at projected marker origin

            # Draw axis too
            cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)

            detected_markers.append((int(ids[i][0]), tvec.flatten()))

        cx = int(self.camera_matrix[0, 2])
        cy = int(self.camera_matrix[1, 2])
        cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)  # Blue dot at optical center
        cv2.imshow("ArUco Detection", color_image)
        cv2.waitKey(1)

        return detected_markers

        # return [(int(ids[i][0]), tvecs[i][0]) for i in range(len(ids))]

    def estimate_pose_single_marker(self, corners, marker_length, camera_matrix, dist_coeffs, z_override=None):
        """
        Manually estimate pose of a single marker using solvePnP.

        Args:
            corners: 4x2 array of marker corners.
            marker_length: Length of the marker's side (in meters).
            camera_matrix: Camera intrinsic matrix.
            dist_coeffs: Distortion coefficients.

        Returns:
            rvec, tvec: Rotation and translation vectors.
        """
        # Define 3D object points of the marker in its own coordinate frame
        half_len = marker_length / 2.0
        obj_points = np.array([
            [-half_len,  half_len, 0],
            [ half_len,  half_len, 0],
            [ half_len, -half_len, 0],
            [-half_len, -half_len, 0]
        ], dtype=np.float32)

        img_points = corners.reshape(-1, 2).astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

        if z_override is not None:
            # Scale the tvec so that its Z matches the depth reading
            scale = z_override / tvec[2][0]
            tvec = tvec * scale

        return rvec, tvec

    def get_translation_of_marker_0(self) -> t.List[float]:
        """
        Returns:
            Translation vector of marker ID 0 in meters: [x, y, z]
        """
        markers = self.find_aruco_codes_in_the_image()
        for marker_id, translation in markers:
            if marker_id == 0:
                return translation.tolist()
        raise ValueError("Marker ID 0 not found.")

    def show_rgb_and_depth(self):
        """
        Display the RGB and depth images side-by-side. 
        Depth image is color mapped for visualisation.
        Hovering the mouse prints depth (in meters) at that point.
        """

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                depth = param['depth_frame'].get_distance(x, y)
                print(f"Depth at ({x},{y}): {depth:.3f} m")

        color_image, depth_frame = self.get_image()

        # Convert depth to color map
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Concatenate RGB and Depth side by side
        images_combined = np.hstack((color_image, depth_colormap))

        # Create interactive window
        cv2.namedWindow("RGB + Depth")
        cv2.setMouseCallback("RGB + Depth", mouse_callback, {'depth_frame': depth_frame})

        cv2.imshow("RGB + Depth", images_combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = CameraOperations()
    while True:
        try:
            x, y, z = cam.get_translation_of_marker_0()
            print(f"Marker location: x={x:.3f}, y={y:.3f}, z={z:.3f} m")
        except ValueError:
            print("Marker ID 0 not detected.")

# if __name__ == "__main__":
#     cam = CameraOperations()
#     cam.show_rgb_and_depth()
