import typing as t
import pyrealsense2 as rs
import numpy as np
import cv2


def convert_depth_to_phys_coord_using_realsense(x, y, depth, camera_matrix, dist_coeffs, width, height):
    """
    Convert depth and pixel coordinates to 3D physical coordinates using RealSense intrinsics.

    Args:
        x (int): The x-coordinate (pixel) of the point in the image.
        y (int): The y-coordinate (pixel) of the point in the image.
        depth (float): The depth value at the given pixel (in meters).
        camera_matrix (np.ndarray): Camera intrinsic matrix (K).
        dist_coeffs (np.ndarray): Distortion coefficients (D).
        width (int): Width of the camera frame.
        height (int): Height of the camera frame.

    Returns:
        tuple: A tuple (X, Y, Z) representing the 3D coordinates in the camera frame.
               X (float): The physical X-coordinate (right).
               Y (float): The physical Y-coordinate (down).
               Z (float): The physical Z-coordinate (forward).
    """
    # Parse the camera intrinsics
    intrinsics = rs.intrinsics()
    intrinsics.width = width
    intrinsics.height = height
    intrinsics.ppx = camera_matrix[0, 2]  # Principal point x (cx)
    intrinsics.ppy = camera_matrix[1, 2]  # Principal point y (cy)
    intrinsics.fx = camera_matrix[0, 0]  # Focal length x (fx)
    intrinsics.fy = camera_matrix[1, 1]  # Focal length y (fy)
    intrinsics.model = rs.distortion.none  # Assuming no distortion model
    intrinsics.coeffs = [i for i in dist_coeffs]  # Distortion coefficients

    # Use RealSense SDK to deproject pixel to 3D point
    result = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)

    # RealSense output [right (X), down (Y), forward (Z)].
    return result[0], result[1], result[2]


class CameraOperations:
    def __init__(self):
        """
        Initialize the RealSense D435 camera and align depth to color stream.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        self.marker_length = 0.05  # [m] physical marker size

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

        max_width = 1280
        max_height = 720
        re_color_image = cv2.resize(color_image, (max_width, max_height))
        cv2.imshow("ArUco Detection", re_color_image)
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

    def find_tennis(self):

        color_image, depth = self.get_image()

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Convert to HSV for color-based filtering
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Define HSV range for the green/yellow ball
        lower_yellow = np.array([40, 100, 150])  # Lower bound
        upper_yellow = np.array([90, 255, 255])  # Upper bound

        # Create a binary mask
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Apply morphological closing (dilate -> erode) to fill holes
        kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Resize mask to show it
        mask_resized = cv2.resize(mask_closed, (1280, 720))
        cv2.imshow("Mask", mask_resized)

        # Apply bitwise AND to isolate the green/yellow parts of the image
        color_filtered = cv2.bitwise_and(gray, gray, mask=mask_closed)

        # Use HoughCircles to detect circles in the filtered image
        circles = cv2.HoughCircles(
            color_filtered,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=30,  # Minimum radius of the circles
            maxRadius=60  # Maximum radius of the circles (adjust based on actual size)
        )

        if circles is not None:
            print(f"Pixel coordinate of tennis ball: {circles[0][0]}")
            circles = np.uint16(np.around(circles))  # Round the values
            for i in circles[0, :]:
                # Draw the circle
                cv2.circle(color_image, (i[0], i[1]), i[2], (100, 255, 0), 3)
                # Draw the circle's center
                cv2.circle(color_image, (i[0], i[1]), 2, (0, 0, 255), 3)

                # Display the original image with detected circles
                # Resize image to show it
                #img_resized = cv2.resize(color_image, (1280, 720))
                #cv2.imshow("Detected Tennis Balls", img_resized)

                # Access the first ball's pixel coordinate
                ball_pixel_coord = circles[0][0]  # The first detected circle
                u, v = ball_pixel_coord[0], ball_pixel_coord[1]  # Pixel coordinates of the circle's center

                # Get dimensions of the depth frame
                depth_width = depth.get_width()
                depth_height = depth.get_height()
                print(f"Depth frame dimensions: {depth_width} x {depth_height}")

                # Get dimensions of the color frame (image resolution from the HoughCircles detection)
                color_height, color_width = color_image.shape[:2]
                print(f"Color frame dimensions: {color_width} x {color_height}")

                # Rescale the circle's coordinates (u, v) if resolutions differ
                u_rescaled = int(u * (depth_width / color_width))
                v_rescaled = int(v * (depth_height / color_height))
                print(f"Rescaled pixel coordinate of ball center: ({u_rescaled}, {v_rescaled})")

                # Get depth value using the rescaled coordinates
                depth_value = depth.get_distance(u_rescaled, v_rescaled)  # Depth at the new scaled coordinates
                print(f"Depth at ball center (meters): {depth_value}")

                # Load camera intrinsics from RealSense
                profile = self.pipeline.get_active_profile()
                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intrinsics = color_stream.get_intrinsics()
                self.camera_matrix = np.array([
                    [intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1]
                ])
                self.dist_coeffs = np.array(intrinsics.coeffs)

                width = color_width
                height = color_height

                result = convert_depth_to_phys_coord_using_realsense(u_rescaled, v_rescaled, depth_value, self.camera_matrix,
                                                                          self.dist_coeffs, width, height)
                print(f"Real-world coordinates of ball center (meters): {result}")

        else:
            print("No circles detected.")

if __name__ == "__main__":
    cam = CameraOperations()
    while True:
        try:
            x, y, z = cam.get_translation_of_marker_0()
            print(f"Marker location: x={x:.3f}, y={y:.3f}, z={z:.3f} m")
            cam.find_tennis()
        except ValueError:
            print("Marker ID 0 not detected.")

# if __name__ == "__main__":
#     cam = CameraOperations()
#     cam.show_rgb_and_depth()
