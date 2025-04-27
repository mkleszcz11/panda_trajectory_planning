import math
import typing as t
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import pyrealsense2 as rs
except ImportError:
    print("RealSense SDK not found. Camera operations will be limited to hardcoded intrinsics.")
    rs = None


class CameraOperations:
    def __init__(self):
        """
        Initialize the RealSense D435i camera and align depth to color stream.
        If no camera is present (USE_REALSENSE = False), fallback to hardcoded intrinsics.
        """
        self.USE_REALSENSE = False  # Toggle to False if no camera is connected

        if self.USE_REALSENSE:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

            try:
                self.pipeline.start(config)
                align_to = rs.stream.color
                self.align = rs.align(align_to)

                profile = self.pipeline.get_active_profile()
                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = color_stream.get_intrinsics()

                self.camera_matrix = np.array([
                    [intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1]
                ])
                self.dist_coeffs = np.array(intr.coeffs)

            except Exception as e:
                print("Could not start RealSense camera, falling back to defaults.")
                self.USE_REALSENSE = False
                self.pipeline = None
                self._use_default_intrinsics()

        else:
            self.pipeline = None
            self._use_default_intrinsics()

        # ArUco dictionary and detector
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.marker_length = 0.05  # meters

    def _use_default_intrinsics(self):
        """
        Use default intrinsics for D435i @ 1920x1080.
        """
        self.camera_matrix = np.array([
            [1380.0, 0, 960.0],
            [0, 1380.0, 540.0],
            [0, 0, 1.0]
        ])
        self.dist_coeffs = np.zeros(5)

    def get_image(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Capture one color + depth frame.

        Returns:
            color_image: BGR image
            depth_frame: aligned depth frame
        """
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                raise RuntimeError("Could not get frames from RealSense camera")

            color_image = np.asanyarray(color_frame.get_data())
            return color_image, depth_frame
        except:
            # If RealSense is not available, use a reference image
            this_file_dir = os.path.dirname(os.path.abspath(__file__))
            reference_image_path = os.path.join(this_file_dir, "reference_image.png")
            color_image = cv2.imread(reference_image_path)
            # Assume depth is 1.2 meters for all pixels
            depth_frame = np.full(color_image.shape[:2], 1.2, dtype=np.float32)
            return color_image, depth_frame


    def find_aruco_codes_in_the_image(self) -> t.List[t.Tuple[int, np.ndarray, np.ndarray]]:
        """
        Detect ArUco markers and return their translation and rotation vectors.

        Returns:
            List of (marker_id, tvec, rvec)
        """
        color_image, depth_frame = self.get_image()
        corners, ids, _ = self.detector.detectMarkers(color_image)
        if ids is None:
            return []

        detected_markers = []

        for i, corner in enumerate(corners):
            marker_id = int(ids[i][0])

            # Marker center for depth lookup
            cx = int(corner[0][:, 0].mean())
            cy = int(corner[0][:, 1].mean())

            marker_id = int(ids[i][0])
            cv2.putText(
                color_image,
                f"ID: {marker_id}",
                (cx + 10, cy - 10),  # slightly offset from the center
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,                # font scale
                (0, 255, 0),        # green text
                2,                  # thickness
                cv2.LINE_AA
            )

            if self.USE_REALSENSE:
                z_depth = depth_frame.get_distance(cx, cy)
            else:
                z_depth = 1.2  # fallback

            # Estimate pose
            rvec, tvec = self.estimate_pose_single_marker(
                corner, self.marker_length, self.camera_matrix, self.dist_coeffs, z_override=z_depth
            )
            tvec[2][0] = z_depth  # enforce correct Z

            detected_markers.append((marker_id, tvec.flatten(), rvec.flatten()))

            # Visualization
            cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.23)
            cv2.putText(color_image, f"ID: {marker_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Put green dot at the center of the image
        width = color_image.shape[1]
        height = color_image.shape[0]
        cv2.circle(color_image, (width//2, height//2), 5, (0, 255, 0), -1)

        # cv2.imshow("ArUco Detection", color_image)
        # Show it in scale
        cv2.imshow("ArUco Detection", cv2.resize(color_image, (1280, 720)))
        cv2.waitKey(0)

        return detected_markers

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
        # print(f"PNP RETURNING rvec =\n{rvec}, tvec =\n{tvec}")

        if z_override is not None:
            # Scale the tvec so that its Z matches the depth reading
            scale = z_override / tvec[2][0]
            tvec = tvec * scale

        # =====================
        # DEBUG INFO
        # =====================
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        return rvec, tvec

    def get_marker_transforms(self) -> t.Dict[str, np.ndarray]:
        """
        Get full 4x4 transformation matrices from each marker to camera.
        We are assuming rotation only around te Z axis.

        Returns:
            Dict: {"corner_0": T_4x4, "corner_1": T_4x4, ...}
        """
        print("Getting marker transforms...")
        transforms = {}
        markers = self.find_aruco_codes_in_the_image()

        for marker_id, tvec, rvec in markers:
            R_mat, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to matrix

            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = np.asarray(tvec).flatten()

            transforms[f"corner_{marker_id}"] = T

        # Sort by marker ID before returning
        transforms = dict(sorted(transforms.items(), key=lambda item: int(item[0].split("_")[1])))
        print(f"Detected markers: {transforms.keys()}")

        # Print the transformation matrices
        for marker_id, T in transforms.items():
            print(f"Marker {marker_id} transform:\n{T}")

        # =======================
        # 3D Visualization
        # =======================
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Detected Marker Positions (Camera Frame)")

        for corner_name, T in transforms.items():
            # pos = T[:3, 3]
            
            # # Draw as orange spheres
            # ax.scatter(*pos, s=100, color='orange', label=corner_name)
            # ax.text(*pos, corner_name, fontsize=9, color='black')
            origin = T[:3, 3]
            R = T[:3, :3]

            # Frame axes (x=red, y=green, z=blue)
            ax.quiver(*origin, *R[:, 0], length=0.15, color='r')  # X
            ax.quiver(*origin, *R[:, 1], length=0.15, color='g')  # Y
            ax.quiver(*origin, *R[:, 2], length=0.15, color='b')  # Z

            ax.text(*origin, corner_name, fontsize=9, color='k')

        # Draw camera coordinate frame at origin
        ax.quiver(0, 0, 0, 0.25, 0, 0, color='r', linestyle='dashed', label='Cam X')
        ax.quiver(0, 0, 0, 0, 0.25, 0, color='g', linestyle='dashed', label='Cam Y')
        ax.quiver(0, 0, 0, 0, 0, 0.25, color='b', linestyle='dashed', label='Cam Z')
        ax.text(0, 0, 0.05, "Camera Frame", fontsize=10, color='black')

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        # Equal aspect ratio
        max_range = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d()
        ]).ptp(axis=1).max() / 2.0

        mid_x = np.mean(ax.get_xlim3d())
        mid_y = np.mean(ax.get_ylim3d())
        mid_z = np.mean(ax.get_zlim3d())

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=30, azim=60)
        plt.tight_layout()
        # plt.show()

        return transforms

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
        lower_yellow = np.array([40, 100, 130])  # Lower bound
        upper_yellow = np.array([90, 255, 255])  # Upper bound

        # Create a binary mask
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Apply morphological closing (dilate -> erode) to fill holes
        kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Resize mask to show it
        # mask_resized = cv2.resize(mask_closed, (1920, 1080))
        # cv2.imshow("Mask", mask_resized)
        cv2.imshow("Mask", mask_closed)
        # cv2.waitKey(0)

        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask_closed)

        min_area = math.pi*10*10

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                    cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)


        # Apply bitwise AND to isolate the green/yellow parts of the image
        color_filtered = cv2.bitwise_and(gray, gray, mask=clean_mask)

        # Apply Gaussian blur to the binary image to create edge gradients
        blurred = cv2.GaussianBlur(clean_mask, (9, 9), 2)

        cv2.imshow("Filtered", blurred)
        # cv2.waitKey(0)

        # Apply Hough Circle Transform on the blurred mask
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,     # Higher threshold for Canny edge detector
            param2=25,     # Lower this to detect more circles (sensitivity)
            minRadius=30,  # Adjusted based on your mask
            maxRadius=50
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))  # Round the values
            largest_circle = max(circles[0, :], key=lambda c: c[2])
            x, y, r = largest_circle
            print(f"Largest circle: center=({x}, {y}), radius={r}")

            # Draw the circle
            cv2.circle(color_image, (x, y), r, (100, 255, 0), 3)
            # Draw the circle's center
            cv2.circle(color_image, (x, y), 2, (0, 0, 255), 3)

            # Display the original image with detected circles
            # Resize image to show it
            # img_resized = cv2.resize(color_image, (1280, 720))
            cv2.imshow("Detected Tennis Balls", color_image)
            cv2.waitKey(1)
            cv2.destroyAllWindows()

            # Access the first ball's pixel coordinate
            u, v = x, y  # Pixel coordinates of the circle's center

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

            # REMOVE REMOVE
            depth_value_kek = depth.get_distance(x, y)  # Depth at the new scaled coordinates
            print(f"Depth at ball center (meters) KEK: {depth_value_kek}")
            # REMOVE REMOVE

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

            result = self.convert_depth_to_phys_coord_using_realsense(u_rescaled, v_rescaled, depth_value, self.camera_matrix,
                                                                        self.dist_coeffs, width, height)
            print(f"Real-world coordinates of ball center (meters): {result}")
            return True, result[0], result[1], result[2]
        else:
            print("No circles detected.")
            return False, 0, 0, 0

    def convert_depth_to_phys_coord_using_realsense(self, x, y, depth, camera_matrix, dist_coeffs, width, height):
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

# if __name__ == "__main__":
#     cam = CameraOperations()
#     while True:
#         try:
#             x, y, z = cam.get_corners_translations()
#             print(f"Marker location: x={x:.3f}, y={y:.3f}, z={z:.3f} m")
#         except ValueError:
#             print("Marker ID 0 not detected.")

# if __name__ == "__main__":
#     cam = CameraOperations()
#     while True:
#         try:
#             transforms = cam.get_marker_transforms()
#             for marker_id, T in transforms.items():
#                 print(f"Marker {marker_id} transform:\n{T}")
#             # x,y,z = cam.find_tennis()
#         except ValueError:
#             print("Marker ID 0 not detected.")

# if __name__ == "__main__":
#     cam = CameraOperations()
#     while True:
#         try:
#             cam.find_aruco_codes_in_the_image()
#             # print(f"Marker location: x={x:.3f}, y={y:.3f}, z={z:.3f} m")
#             # x,y,z = cam.find_tennis()
#         except ValueError:
#             print("Some marker was not detected.")

if __name__ == "__main__":
    cam = CameraOperations()
    cam.show_rgb_and_depth()
