import math
import os
import typing as t
import cv2
import matplotlib.pyplot as plt # Keep for ArUco visualization if re-enabled
import numpy as np
import torch
from ultralytics import YOLO
from mpl_toolkits.mplot3d import Axes3D


import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
import math
import torch
from klemol_planner.camera_utils.capture_realsense_frame_yolo import setup_realsense_pipeline, get_aligned_frames, save_frame
from klemol_planner.camera_utils.antipodal_grasp_planner2 import AntipodalGraspPlanner


# --- Static method for getting depth ---
def convert_depth_to_phys_coord_using_realsense_intrinsics(x, y, depth, intrinsics_obj):
    if intrinsics_obj is None or depth <= 0: return 0.0, 0.0, 0.0
    try:
        x = int(max(0, min(x, intrinsics_obj.width - 1)))
        y = int(max(0, min(y, intrinsics_obj.height - 1)))
        result = rs.rs2_deproject_pixel_to_point(intrinsics_obj, [x, y], depth)
        return result[0], result[1], result[2]
    except Exception:
        return 0.0, 0.0, 0.0



class CameraOperations:
    def __init__(self):
        """
        Initialize the RealSense D435i camera and align depth to color stream.
        If no camera is present (USE_REALSENSE = False), fallback to hardcoded intrinsics.
        """
        self.USE_REALSENSE = True  # Toggle to False if no camera is connected

        if self.USE_REALSENSE:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

            # try:
            self.pipeline.start(config)
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            profile = self.pipeline.get_active_profile()
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            self.color_width = intr.width
            self.color_height = intr.height

            self.camera_matrix = np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]
            ])
            self.dist_coeffs = np.array(intr.coeffs)

            # except Exception as e:
            #     print("Could not start RealSense camera, falling back to defaults.")
            #     self.USE_REALSENSE = False
            #     self.pipeline = None
            #     self._use_default_intrinsics()

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


    def _get_rs_intrinsics_object(self) -> t.Optional[rs.intrinsics]:
        if not rs: return None
        if self.USE_REALSENSE and self.pipeline:
            try:
                profile = self.pipeline.get_active_profile()
                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                return color_stream.get_intrinsics()
            except Exception:  # Fall through to stored if live fails
                pass

        if self.camera_matrix is None: return None
        intr = rs.intrinsics()
        intr.width = self.color_width
        intr.height = self.color_height
        intr.ppx = self.camera_matrix[0, 2]
        intr.ppy = self.camera_matrix[1, 2]
        intr.fx = self.camera_matrix[0, 0]
        intr.fy = self.camera_matrix[1, 1]
        intr.model = rs.distortion.none  # Assuming no/minimal distortion or already undistorted
        intr.coeffs = list(self.dist_coeffs) if self.dist_coeffs is not None else [0.0] * 5
        return intr


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


    def get_yolo_grasp_for_object(self, target_label: str, yolo_model,
                                  confidence_thresh=0.25, visualize: bool = False) \
            -> t.Tuple[t.Optional[np.ndarray], t.Optional[float]]:
        """
        Detects an object, plans a grasp, returns 3D center and 2D yaw (in camera XY plane).

        Args:
            target_label (str): Object label (e.g., "banana").
            yolo_model: Pre-loaded YOLOv8 segmentation model.
            confidence_thresh (float): Min confidence for YOLO.
            visualize (bool): If True, displays intermediate steps.

        Returns:
            A tuple (grasp_center_3d, grasp_yaw_radians_cam_xy):
            - grasp_center_3d: np.array([X, Y, Z]) in camera frame (meters).
                               X right, Y down, Z "forward" (your convention is -Z for workspace forward).
            - grasp_yaw_radians_cam_xy: float, yaw angle of the grasp line (p1 to p2)
                                        in the camera's XY plane, relative to camera X-axis.
            Returns (None, None) if not found or error.
        """
        color_image, depth_frame_obj = self.get_image()
        if color_image is None or depth_frame_obj is None:
            return None, None

        yolo_results = yolo_model.predict(color_image, verbose=False,
                                          device='cuda' if torch.cuda.is_available() else 'cpu')
        result = yolo_results[0]

        if result.masks is None or result.boxes is None:
            if visualize: cv2.imshow("YOLO Grasp (No Detections)", cv2.resize(color_image, (
            self.color_width // 2, self.color_height // 2))); cv2.waitKey(1)
            return None, None

        binary_mask_target = None
        target_box = None  # For visualization
        for i in range(len(result.boxes)):
            conf = result.boxes.conf[i].item()
            cls_id = int(result.boxes.cls[i].item())
            label = result.names[cls_id]
            if label == target_label and conf >= confidence_thresh:
                mask_raw = result.masks.data[i].cpu().numpy()
                binary_mask_target = cv2.resize(mask_raw, (self.color_width, self.color_height),
                                                interpolation=cv2.INTER_NEAREST)
                binary_mask_target = (binary_mask_target > 0.5).astype(np.uint8) * 255
                target_box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                break

        if binary_mask_target is None:
            if visualize:
                cv2.putText(color_image, f"'{target_label}' not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.imshow("YOLO Grasp (Target Not Found)",
                           cv2.resize(color_image, (self.color_width // 2, self.color_height // 2)));
                cv2.waitKey(1)
            return None, None

        planner = AntipodalGraspPlanner(
            max_gripper_opening_px=120, min_grasp_width_px=10,
            angle_tolerance_deg=10, contour_approx_epsilon_factor=0.004,
            normal_neighborhood_k=5, dist_penalty_weight=0.03, width_favor_narrow_weight=0.02
        )
        local_grasps, object_centroid_px = planner.find_grasps(binary_mask_target)

        if not local_grasps or object_centroid_px is None:
            if visualize:
                cv2.rectangle(color_image, (target_box[0], target_box[1]), (target_box[2], target_box[3]), (0, 255, 0),
                              1)
                cv2.putText(color_image, f"'{target_label}' (No Grasps)", (target_box[0], target_box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                cv2.imshow("YOLO Grasp (No Grasps Found)",
                           cv2.resize(color_image, (self.color_width // 2, self.color_height // 2)));
                cv2.waitKey(1)
            return None, None

        best_local_grasp = local_grasps[0]
        abs_grasp = planner.transform_grasp_to_image_space(best_local_grasp, object_centroid_px)
        if not abs_grasp: return None, None

        rs_intrinsics = self._get_rs_intrinsics_object()
        if rs_intrinsics is None: return None, None

        points_3d = {}
        for name, pt2d_tuple in [('p1', abs_grasp['p1']), ('p2', abs_grasp['p2'])]:
            u, v = int(pt2d_tuple[0]), int(pt2d_tuple[1])
            u_c = max(0, min(u, self.color_width - 1));
            v_c = max(0, min(v, self.color_height - 1))
            depth_val = depth_frame_obj.get_distance(u_c, v_c) if rs and isinstance(depth_frame_obj, rs.frame) else \
            depth_frame_obj[v_c, u_c]
            if depth_val <= 0:
                points_3d[name] = np.array([0, 0, 0])
            else:
                points_3d[name] = np.array(
                    convert_depth_to_phys_coord_using_realsense_intrinsics(u, v, depth_val, rs_intrinsics))

        p1_3d, p2_3d = points_3d['p1'], points_3d['p2']
        if p1_3d[2] <= 1e-3 or p2_3d[2] <= 1e-3:  # Check Z-depth validity
            if visualize:
                vis_img_fail = planner.visualize_grasps(color_image, [abs_grasp], 1,
                                                        object_centroid_abs=object_centroid_px)
                cv2.putText(vis_img_fail, "Invalid 3D points for grasp", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.imshow("YOLO Grasp", cv2.resize(vis_img_fail, (self.color_width // 2, self.color_height // 2)));
                cv2.waitKey(1)
            return None, None

        grasp_center_3d = (p1_3d + p2_3d) / 2.0

        # Calculate Yaw: Angle of the line (p1_3d to p2_3d) projected onto camera's XY plane,
        # relative to camera's X-axis.
        # Camera X-axis: [1,0,0], Camera Y-axis: [0,1,0]
        # Vector representing grasp width direction in 3D (from p1 to p2)
        grasp_vector_3d = p2_3d - p1_3d

        # Project this vector onto camera's XY plane (i.e., take its X and Y components)
        dx_cam = grasp_vector_3d[0]  # X component in camera frame
        dy_cam = grasp_vector_3d[1]  # Y component in camera frame

        # Yaw is the angle of this (dx_cam, dy_cam) vector
        grasp_yaw_radians_cam_xy = math.atan2(dy_cam, dx_cam)

        if visualize:
            vis_img = planner.visualize_grasps(color_image, [abs_grasp], 1, object_centroid_abs=object_centroid_px)
            cv2.putText(vis_img, f"'{target_label}' Grasp Found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2)
            cv2.putText(vis_img,
                        f"Center 3D:({grasp_center_3d[0]:.2f},{grasp_center_3d[1]:.2f},{grasp_center_3d[2]:.2f})m",
                        (10, vis_img.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 50), 1)
            cv2.putText(vis_img, f"Yaw (camXY): {math.degrees(grasp_yaw_radians_cam_xy):.1f} deg",
                        (10, vis_img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 50), 1)

            # Draw the grasp line used for yaw calculation in 2D
            p1_2d_tuple = tuple(map(int, abs_grasp['p1']))
            p2_2d_tuple = tuple(map(int, abs_grasp['p2']))
            cv2.arrowedLine(vis_img, p1_2d_tuple, p2_2d_tuple, (255, 100, 255), 2, tipLength=0.05)

            cv2.imshow("YOLO Grasp", cv2.resize(vis_img, (self.color_width // 2, self.color_height // 2)))
            cv2.waitKey(0)

        return grasp_center_3d, grasp_yaw_radians_cam_xy


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
            cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
            cv2.putText(color_image, f"ID: {marker_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add Box labels for markers 10 and 11
            if marker_id == 10:
                cv2.putText(color_image, "Box 1", (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif marker_id == 11:
                cv2.putText(color_image, "Box 2", (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Put green dot and crosshairs at the center of the image
        width = color_image.shape[1]
        height = color_image.shape[0]
        cv2.circle(color_image, (width//2, height//2), 5, (0, 255, 0), -1)

        cv2.drawMarker(color_image, (width//2, height//2), color=(0, 255, 0), thickness=5,
                       markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA,
                       markerSize=50)

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
            cv2.waitKey(0)
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

    @staticmethod
    def _convert_depth_to_phys_coord_using_realsense_intrinsics(x, y, depth, intrinsics):
        if intrinsics is None or depth <= 0: return 0.0, 0.0, 0.0
        try:
            x = int(max(0, min(x, intrinsics.width - 1)))
            y = int(max(0, min(y, intrinsics.height - 1)))
            result = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
            return result[0], result[1], result[2]
        except Exception:
            return 0.0, 0.0, 0.0

    def find_grasp(self, object_to_find: str = "banana", timeout: float = 10.0) -> t.List:
        """
        TODO
        Return success, x, y, z, rotation
        """
        pipeline = self.pipeline
        profile = self.pipeline.get_active_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

        if pipeline is None: exit()

        color_intrinsics = color_profile.get_intrinsics()
        actual_format = color_profile.format()
        print(
            f"Pipeline started. Color Format: {actual_format}, Intrinsics: W={color_intrinsics.width}, H={color_intrinsics.height}")

        if not torch.cuda.is_available():
            print("!!! WARNING: CUDA is not available. Running on CPU. !!!")
            device = 'cpu'
        else:
            print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            device = 'cuda'

        print(f"Loading YOLOv8 Segmentation model onto {device.upper()}...")
        try:
            model = YOLO('yolov8l-seg.pt')  # Or your preferred model
            model.to(device)
            coco_names = model.names
            print("YOLOv8 model loaded.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}.")
            if pipeline: pipeline.stop()
            exit()

        print("Initializing Antipodal Grasp Planner...")
        GRASP_MAX_OPENING_PIXELS = 120
        GRASP_MIN_WIDTH_PIXELS = 10
        GRASP_ANGLE_TOLERANCE_DEGREES = 5  # Increased for curved objects
        GRASP_CONTOUR_EPSILON_FACTOR = 0.0002  # More detail for curves
        GRASP_NORMAL_NEIGHBORHOOD_K = 10  # Larger k for more detailed contours
        GRASP_DIST_PENALTY_WEIGHT = 0.03
        GRASP_WIDTH_FAVOR_NARROW_WEIGHT = 0.02  # Favor slightly narrower for elongated

        grasp_planner = AntipodalGraspPlanner(
            max_gripper_opening_px=GRASP_MAX_OPENING_PIXELS,
            min_grasp_width_px=GRASP_MIN_WIDTH_PIXELS,
            angle_tolerance_deg=GRASP_ANGLE_TOLERANCE_DEGREES,
            contour_approx_epsilon_factor=GRASP_CONTOUR_EPSILON_FACTOR,
            normal_neighborhood_k=GRASP_NORMAL_NEIGHBORHOOD_K,
            dist_penalty_weight=GRASP_DIST_PENALTY_WEIGHT,
            width_favor_narrow_weight=GRASP_WIDTH_FAVOR_NARROW_WEIGHT
        )
        print("Antipodal Grasp Planner initialized.")
        TARGET_CLASS_NAME = "banana"  # Change to your target: "cell phone", "cup", etc.

        confidence_threshold = 0.5
        output_folder = f"live_{TARGET_CLASS_NAME.replace(' ', '_')}_grasps_tracked"
        np.random.seed(42)  # Consistent colors
        mask_colors = np.random.randint(0, 256, (len(coco_names), 3), dtype=np.uint8)

        # --- Dictionary to store best grasps for tracked objects ---
        tracked_object_best_local_grasps = {}  # Key: track_id, Value: best local grasp dict found so far
        # -------------------------------------------------------------

        print(f"\nStarting live detection, tracking, and grasping for '{TARGET_CLASS_NAME}'... Press 'q' to quit.")
        fps, frame_count, start_time = 0.0, 0, time.time()

        is_grasp_candidate_found = False
        start_time = time.time()

        while not is_grasp_candidate_found:
            curr_time = time.time()
            if curr_time - start_time > timeout:
                break

            color_image_from_realsense, depth_frame = get_aligned_frames(pipeline, align_to=rs.stream.color)
            if color_image_from_realsense is None or depth_frame is None:
                time.sleep(0.01)
                continue

            color_image_bgr = color_image_from_realsense
            draw_image = color_image_bgr.copy()
            overlay = draw_image.copy()

            # --- Perform Object Detection, Segmentation & TRACKING ---
            # Using model.track() for object tracking
            results = model.track(color_image_bgr, persist=True, tracker="botsort.yaml", verbose=False)
            # For ByteTrack: tracker="bytetrack.yaml"
            # `persist=True` tells YOLO to remember tracks between frames.
            # ------------------------------------------------------

            current_frame_grasps_to_visualize = []
            current_frame_target_centroids = {}  # Store centroids for visualization {track_id: centroid}

            if results and results[0].boxes is not None and results[0].masks is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                masks_data_raw = results[0].masks.data.cpu().numpy()

                # --- Get Tracking IDs ---
                track_ids = None
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                # ------------------------

                current_img_shape = (draw_image.shape[0], draw_image.shape[1])

                for i in range(len(boxes)):
                    if confs[i] < confidence_threshold: continue

                    x1, y1, x2, y2 = map(int, boxes[i])
                    cls_id = int(clss[i])
                    class_name = coco_names[cls_id]
                    track_id = track_ids[i] if track_ids is not None else -1  # Use -1 if no track ID

                    # Draw BBox and label
                    label_text = f"ID:{track_id} {class_name} {confs[i]:.2f}" if track_id != -1 else f"{class_name} {confs[i]:.2f}"
                    cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(draw_image, label_text, (x1, y1 - 10 if y1 > 20 else y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    individual_mask_raw = masks_data_raw[i]
                    if individual_mask_raw.shape != current_img_shape:
                        individual_mask_resized = cv2.resize(individual_mask_raw,
                                                             (current_img_shape[1], current_img_shape[0]),
                                                             interpolation=cv2.INTER_NEAREST)
                    else:
                        individual_mask_resized = individual_mask_raw
                    binary_mask_for_planner = (individual_mask_resized > 0.5).astype(np.uint8) * 255

                    mask_for_overlay = binary_mask_for_planner.astype(bool)
                    overlay[mask_for_overlay] = mask_colors[cls_id].tolist()

                    if class_name == TARGET_CLASS_NAME and track_id != -1:  # Only process tracked target objects
                        new_local_grasps, current_obj_centroid = grasp_planner.find_grasps(
                            binary_mask_for_planner.copy())

                        if current_obj_centroid is not None:
                            current_frame_target_centroids[track_id] = current_obj_centroid

                        best_new_local_grasp_this_frame = None
                        if new_local_grasps:
                            best_new_local_grasp_this_frame = new_local_grasps[0]  # Highest score from current frame

                            # --- Update stored best grasp for this track_id ---
                            if track_id not in tracked_object_best_local_grasps:
                                tracked_object_best_local_grasps[track_id] = best_new_local_grasp_this_frame
                                print(
                                    f"  New track ID {track_id} ({TARGET_CLASS_NAME}): Storing initial best grasp (Score: {best_new_local_grasp_this_frame['score']:.3f})")
                            else:
                                stored_grasp = tracked_object_best_local_grasps[track_id]
                                if best_new_local_grasp_this_frame['score'] > stored_grasp['score']:
                                    tracked_object_best_local_grasps[track_id] = best_new_local_grasp_this_frame
                                    print(
                                        f"  Track ID {track_id} ({TARGET_CLASS_NAME}): Updated best grasp (New Score: {best_new_local_grasp_this_frame['score']:.3f} > Old Score: {stored_grasp['score']:.3f})")
                            # --------------------------------------------------

                        # --- Use the (potentially updated) stored best grasp for visualization and 3D ---
                        if track_id in tracked_object_best_local_grasps and current_obj_centroid is not None:
                            stable_local_grasp_to_use = tracked_object_best_local_grasps[track_id]
                            abs_grasp_to_visualize = grasp_planner.transform_grasp_to_image_space(
                                stable_local_grasp_to_use,
                                current_obj_centroid)

                            if abs_grasp_to_visualize:
                                current_frame_grasps_to_visualize.append(abs_grasp_to_visualize)

                                # --- Convert this stable grasp to 3D (example: for the first target object with a grasp) ---
                                if len(current_frame_grasps_to_visualize) == 1:  # For simplicity, only print 3D for one
                                    p1_2d, p2_2d, center_2d = abs_grasp_to_visualize['p1'], abs_grasp_to_visualize[
                                        'p2'], \
                                        abs_grasp_to_visualize['center_px']
                                    depth_w, depth_h = depth_frame.get_width(), depth_frame.get_height()
                                    d1 = depth_frame.get_distance(p1_2d[0] % depth_w,
                                                                  p1_2d[1] % depth_h) if depth_frame else 0
                                    d2 = depth_frame.get_distance(p2_2d[0] % depth_w,
                                                                  p2_2d[1] % depth_h) if depth_frame else 0
                                    dc = depth_frame.get_distance(center_2d[0] % depth_w,
                                                                  center_2d[1] % depth_h) if depth_frame else 0

                                    p1_3d_m = convert_depth_to_phys_coord_using_realsense_intrinsics(p1_2d[0], p1_2d[1],
                                                                                                     d1,
                                                                                                     color_intrinsics)
                                    p2_3d_m = convert_depth_to_phys_coord_using_realsense_intrinsics(p2_2d[0], p2_2d[1],
                                                                                                     d2,
                                                                                                     color_intrinsics)
                                    width_3d_m = math.sqrt(
                                        sum((c1 - c2) ** 2 for c1, c2 in zip(p1_3d_m, p2_3d_m))) if all(
                                        c != 0 for c in p1_3d_m) and all(c != 0 for c in p2_3d_m) else 0.0
                                    print(
                                        f"    Track ID {track_id} - Stable Grasp 3D (m): P1({p1_3d_m[0]:.3f},{p1_3d_m[1]:.3f},{p1_3d_m[2]:.3f}), "
                                        f"P2({p2_3d_m[0]:.3f},{p2_3d_m[1]:.3f},{p2_3d_m[2]:.3f}), Width: {width_3d_m:.3f}m")
                                    
                                    # --- Draw 2D image-space origin (0,0) and axis lines ---
                                    origin_2d = (0, 0)  # top-left corner of image

                                    # Define axis end points (length = 80 pixels)
                                    x_axis_end = (origin_2d[0] + 80, origin_2d[1])
                                    y_axis_end = (origin_2d[0], origin_2d[1] + 80)

                                    # Draw axes
                                    cv2.arrowedLine(draw_image, origin_2d, x_axis_end, (0, 0, 255), 2, tipLength=0.2)  # X-axis in red
                                    cv2.arrowedLine(draw_image, origin_2d, y_axis_end, (0, 255, 0), 2, tipLength=0.2)  # Y-axis in green

                                    # Draw origin point and labels
                                    cv2.circle(draw_image, origin_2d, 5, (255, 255, 255), -1)  # white dot at (0,0)
                                    cv2.putText(draw_image, "Origin (0,0)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    cv2.putText(draw_image, "X", (x_axis_end[0] + 5, x_axis_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                    cv2.putText(draw_image, "Y", (y_axis_end[0] + 5, y_axis_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                                    # Draw grasp endpoints and center
                                    cv2.circle(draw_image, p1_2d, 5, (0, 255, 0), -1)  # P1 - green
                                    cv2.circle(draw_image, p2_2d, 5, (0, 255, 0), -1)  # P2 - green
                                    cv2.circle(draw_image, center_2d, 5, (255, 0, 0), -1)  # Center - blue

                                    # Label P1 and P2
                                    cv2.putText(draw_image, "P1", (p1_2d[0] + 5, p1_2d[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    cv2.putText(draw_image, "P2", (p2_2d[0] + 5, p2_2d[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    is_grasp_candidate_found = True
                        # -----------------------------------------------------------------------------
            

            cv2.addWeighted(overlay, 0.4, draw_image, 0.6, 0, draw_image)

            if current_frame_grasps_to_visualize:
                # For visualization, we need to pass the *current* centroid of each object if we want to show the CoG dot correctly
                # However, visualize_grasps currently takes only one object_centroid_abs.
                # For simplicity, we'll visualize grasps without showing individual CoGs per grasp in this quick merge.
                # Or, pass the centroid of the first object for which grasps are shown.
                first_tracked_id_with_grasp = None
                if current_frame_grasps_to_visualize:  # Check again
                    # Find a track_id associated with the grasps being visualized
                    # This is a bit tricky if multiple objects are shown. Simplification:
                    if current_frame_target_centroids:
                        first_tracked_id_with_grasp = next(iter(current_frame_target_centroids))

                draw_image = grasp_planner.visualize_grasps(
                    draw_image,
                    current_frame_grasps_to_visualize,
                    num_top_grasps=len(current_frame_grasps_to_visualize),
                    # Show all found stable grasps for tracked targets
                    object_centroid_abs=current_frame_target_centroids.get(
                        first_tracked_id_with_grasp) if first_tracked_id_with_grasp else None
                )

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count, start_time = 0, time.time()
            cv2.putText(draw_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            display_h = 720
            scale_factor = display_h / draw_image.shape[0]
            img_display = cv2.resize(draw_image, (int(draw_image.shape[1] * scale_factor), display_h))
            cv2.imshow(f"Live Tracked {TARGET_CLASS_NAME} Grasping", img_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exit key 'q' pressed.")
                if not os.path.exists(output_folder): os.makedirs(output_folder)
                save_frame(color_image_bgr, output_folder, prefix="last_color")
                save_frame(draw_image, output_folder, prefix="last_detection_grasp_tracked")
                break



        x = (p1_3d_m[0] + p2_3d_m[0]) / 2.0
        y = (p1_3d_m[1] + p2_3d_m[1]) / 2.0
        z = dc

        # Sort points so that the one with the smaller Y (i.e. closer to X-axis) is first
        if abs(p1_3d_m[1]) <= abs(p2_3d_m[1]):
            point_closer_to_x_axis = p1_3d_m
            point_further_to_x_axis = p2_3d_m
        else:
            point_closer_to_x_axis = p2_3d_m
            point_further_to_x_axis = p1_3d_m

        yaw = math.atan2(point_further_to_x_axis[1] - point_closer_to_x_axis[1],
                        point_further_to_x_axis[0] - point_closer_to_x_axis[0])

        print("########################################")
        print("RETURNING GRASP COORD.")
        print("########################################")
        return True, x, y, z, yaw




























# if __name__ == "__main__":

#     cam = CameraOperations()

#     import pyrealsense2 as rs
#     import numpy as np
#     import cv2
#     import os
#     import time
#     from datetime import datetime
#     from ultralytics import YOLO
#     import math
#     import torch

#     # --- Import Custom Modules ---
#     try:
#         from capture_realsense_frame_yolo import setup_realsense_pipeline, get_aligned_frames, save_frame

#         print("Imported capture functions from capture_realsense_frame_yolo.py")
#     except ImportError:
#         print("Error: Could not import from capture_realsense_frame_yolo.py.")
#         exit()

#     try:
#         # Ensure you are using the correct filename if you saved it as antipodal_grasp_planner2.py
#         from antipodal_grasp_planner2 import AntipodalGraspPlanner

#         print("Imported AntipodalGraspPlanner class.")
#     except ImportError:
#         print("Error: Could not import AntipodalGraspPlanner.")
#         print("Please ensure antipodal_grasp_planner2.py (or the correct name) is in the same directory or accessible.")
#         exit()


#     # --- END Import Custom Modules ---

#     def convert_depth_to_phys_coord_using_realsense_intrinsics(x, y, depth, intrinsics):
#         if intrinsics is None or depth <= 0: return 0.0, 0.0, 0.0
#         try:
#             x = int(max(0, min(x, intrinsics.width - 1)))
#             y = int(max(0, min(y, intrinsics.height - 1)))
#             result = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
#             return result[0], result[1], result[2]
#         except Exception:
#             return 0.0, 0.0, 0.0


#     # --- Main Script Logic ---
#     print("Setting up RealSense pipeline...")
#     pipeline, profile, color_profile = setup_realsense_pipeline(request_max_res=False)

#     if pipeline is None: exit()

#     color_intrinsics = color_profile.get_intrinsics()
#     actual_format = color_profile.format()
#     print(
#         f"Pipeline started. Color Format: {actual_format}, Intrinsics: W={color_intrinsics.width}, H={color_intrinsics.height}")

#     if not torch.cuda.is_available():
#         print("!!! WARNING: CUDA is not available. Running on CPU. !!!")
#         device = 'cpu'
#     else:
#         print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
#         device = 'cuda'

#     print(f"Loading YOLOv8 Segmentation model onto {device.upper()}...")
#     try:
#         model = YOLO('yolov8l-seg.pt')  # Or your preferred model
#         model.to(device)
#         coco_names = model.names
#         print("YOLOv8 model loaded.")
#     except Exception as e:
#         print(f"Error loading YOLO model: {e}.")
#         if pipeline: pipeline.stop()
#         exit()

#     print("Initializing Antipodal Grasp Planner...")
#     GRASP_MAX_OPENING_PIXELS = 120
#     GRASP_MIN_WIDTH_PIXELS = 10
#     GRASP_ANGLE_TOLERANCE_DEGREES = 10  # Increased for curved objects
#     GRASP_CONTOUR_EPSILON_FACTOR = 0.004  # More detail for curves
#     GRASP_NORMAL_NEIGHBORHOOD_K = 5  # Larger k for more detailed contours
#     GRASP_DIST_PENALTY_WEIGHT = 0.03
#     GRASP_WIDTH_FAVOR_NARROW_WEIGHT = 0.02  # Favor slightly narrower for elongated

#     grasp_planner = AntipodalGraspPlanner(
#         max_gripper_opening_px=GRASP_MAX_OPENING_PIXELS,
#         min_grasp_width_px=GRASP_MIN_WIDTH_PIXELS,
#         angle_tolerance_deg=GRASP_ANGLE_TOLERANCE_DEGREES,
#         contour_approx_epsilon_factor=GRASP_CONTOUR_EPSILON_FACTOR,
#         normal_neighborhood_k=GRASP_NORMAL_NEIGHBORHOOD_K,
#         dist_penalty_weight=GRASP_DIST_PENALTY_WEIGHT,
#         width_favor_narrow_weight=GRASP_WIDTH_FAVOR_NARROW_WEIGHT
#     )
#     print("Antipodal Grasp Planner initialized.")
#     TARGET_CLASS_NAME = "banana"  # Change to your target: "cell phone", "cup", etc.

#     confidence_threshold = 0.5
#     output_folder = f"live_{TARGET_CLASS_NAME.replace(' ', '_')}_grasps_tracked"
#     np.random.seed(42)  # Consistent colors
#     mask_colors = np.random.randint(0, 256, (len(coco_names), 3), dtype=np.uint8)

#     # --- Dictionary to store best grasps for tracked objects ---
#     tracked_object_best_local_grasps = {}  # Key: track_id, Value: best local grasp dict found so far
#     # -------------------------------------------------------------

#     print(f"\nStarting live detection, tracking, and grasping for '{TARGET_CLASS_NAME}'... Press 'q' to quit.")
#     fps, frame_count, start_time = 0.0, 0, time.time()

#     try:
#         while True:
#             color_image_from_realsense, depth_frame = get_aligned_frames(pipeline, align_to=rs.stream.color)
#             if color_image_from_realsense is None or depth_frame is None:
#                 time.sleep(0.01)
#                 continue

#             color_image_bgr = color_image_from_realsense
#             draw_image = color_image_bgr.copy()
#             overlay = draw_image.copy()

#             # --- Perform Object Detection, Segmentation & TRACKING ---
#             # Using model.track() for object tracking
#             results = model.track(color_image_bgr, persist=True, tracker="botsort.yaml", verbose=False)
#             # For ByteTrack: tracker="bytetrack.yaml"
#             # `persist=True` tells YOLO to remember tracks between frames.
#             # ------------------------------------------------------

#             current_frame_grasps_to_visualize = []
#             current_frame_target_centroids = {}  # Store centroids for visualization {track_id: centroid}

#             if results and results[0].boxes is not None and results[0].masks is not None:
#                 boxes = results[0].boxes.xyxy.cpu().numpy()
#                 confs = results[0].boxes.conf.cpu().numpy()
#                 clss = results[0].boxes.cls.cpu().numpy()
#                 masks_data_raw = results[0].masks.data.cpu().numpy()

#                 # --- Get Tracking IDs ---
#                 track_ids = None
#                 if results[0].boxes.id is not None:
#                     track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#                 # ------------------------

#                 current_img_shape = (draw_image.shape[0], draw_image.shape[1])

#                 for i in range(len(boxes)):
#                     if confs[i] < confidence_threshold: continue

#                     x1, y1, x2, y2 = map(int, boxes[i])
#                     cls_id = int(clss[i])
#                     class_name = coco_names[cls_id]
#                     track_id = track_ids[i] if track_ids is not None else -1  # Use -1 if no track ID

#                     # Draw BBox and label
#                     label_text = f"ID:{track_id} {class_name} {confs[i]:.2f}" if track_id != -1 else f"{class_name} {confs[i]:.2f}"
#                     cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(draw_image, label_text, (x1, y1 - 10 if y1 > 20 else y1 + 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                     individual_mask_raw = masks_data_raw[i]
#                     if individual_mask_raw.shape != current_img_shape:
#                         individual_mask_resized = cv2.resize(individual_mask_raw,
#                                                              (current_img_shape[1], current_img_shape[0]),
#                                                              interpolation=cv2.INTER_NEAREST)
#                     else:
#                         individual_mask_resized = individual_mask_raw
#                     binary_mask_for_planner = (individual_mask_resized > 0.5).astype(np.uint8) * 255

#                     mask_for_overlay = binary_mask_for_planner.astype(bool)
#                     overlay[mask_for_overlay] = mask_colors[cls_id].tolist()

#                     if class_name == TARGET_CLASS_NAME and track_id != -1:  # Only process tracked target objects
#                         new_local_grasps, current_obj_centroid = grasp_planner.find_grasps(
#                             binary_mask_for_planner.copy())

#                         if current_obj_centroid is not None:
#                             current_frame_target_centroids[track_id] = current_obj_centroid

#                         best_new_local_grasp_this_frame = None
#                         if new_local_grasps:
#                             best_new_local_grasp_this_frame = new_local_grasps[0]  # Highest score from current frame

#                             # --- Update stored best grasp for this track_id ---
#                             if track_id not in tracked_object_best_local_grasps:
#                                 tracked_object_best_local_grasps[track_id] = best_new_local_grasp_this_frame
#                                 print(
#                                     f"  New track ID {track_id} ({TARGET_CLASS_NAME}): Storing initial best grasp (Score: {best_new_local_grasp_this_frame['score']:.3f})")
#                             else:
#                                 stored_grasp = tracked_object_best_local_grasps[track_id]
#                                 if best_new_local_grasp_this_frame['score'] > stored_grasp['score']:
#                                     tracked_object_best_local_grasps[track_id] = best_new_local_grasp_this_frame
#                                     print(
#                                         f"  Track ID {track_id} ({TARGET_CLASS_NAME}): Updated best grasp (New Score: {best_new_local_grasp_this_frame['score']:.3f} > Old Score: {stored_grasp['score']:.3f})")
#                             # --------------------------------------------------

#                         # --- Use the (potentially updated) stored best grasp for visualization and 3D ---
#                         if track_id in tracked_object_best_local_grasps and current_obj_centroid is not None:
#                             stable_local_grasp_to_use = tracked_object_best_local_grasps[track_id]
#                             abs_grasp_to_visualize = grasp_planner.transform_grasp_to_image_space(
#                                 stable_local_grasp_to_use,
#                                 current_obj_centroid)

#                             if abs_grasp_to_visualize:
#                                 current_frame_grasps_to_visualize.append(abs_grasp_to_visualize)

#                                 # --- Convert this stable grasp to 3D (example: for the first target object with a grasp) ---
#                                 if len(current_frame_grasps_to_visualize) == 1:  # For simplicity, only print 3D for one
#                                     p1_2d, p2_2d, center_2d = abs_grasp_to_visualize['p1'], abs_grasp_to_visualize[
#                                         'p2'], \
#                                         abs_grasp_to_visualize['center_px']
#                                     depth_w, depth_h = depth_frame.get_width(), depth_frame.get_height()
#                                     d1 = depth_frame.get_distance(p1_2d[0] % depth_w,
#                                                                   p1_2d[1] % depth_h) if depth_frame else 0
#                                     d2 = depth_frame.get_distance(p2_2d[0] % depth_w,
#                                                                   p2_2d[1] % depth_h) if depth_frame else 0
#                                     dc = depth_frame.get_distance(center_2d[0] % depth_w,
#                                                                   center_2d[1] % depth_h) if depth_frame else 0

#                                     p1_3d_m = convert_depth_to_phys_coord_using_realsense_intrinsics(p1_2d[0], p1_2d[1],
#                                                                                                      d1,
#                                                                                                      color_intrinsics)
#                                     p2_3d_m = convert_depth_to_phys_coord_using_realsense_intrinsics(p2_2d[0], p2_2d[1],
#                                                                                                      d2,
#                                                                                                      color_intrinsics)
#                                     width_3d_m = math.sqrt(
#                                         sum((c1 - c2) ** 2 for c1, c2 in zip(p1_3d_m, p2_3d_m))) if all(
#                                         c != 0 for c in p1_3d_m) and all(c != 0 for c in p2_3d_m) else 0.0
#                                     print(
#                                         f"    Track ID {track_id} - Stable Grasp 3D (m): P1({p1_3d_m[0]:.3f},{p1_3d_m[1]:.3f},{p1_3d_m[2]:.3f}), "
#                                         f"P2({p2_3d_m[0]:.3f},{p2_3d_m[1]:.3f},{p2_3d_m[2]:.3f}), Width: {width_3d_m:.3f}m")
#                         # -----------------------------------------------------------------------------

#             cv2.addWeighted(overlay, 0.4, draw_image, 0.6, 0, draw_image)

#             if current_frame_grasps_to_visualize:
#                 # For visualization, we need to pass the *current* centroid of each object if we want to show the CoG dot correctly
#                 # However, visualize_grasps currently takes only one object_centroid_abs.
#                 # For simplicity, we'll visualize grasps without showing individual CoGs per grasp in this quick merge.
#                 # Or, pass the centroid of the first object for which grasps are shown.
#                 first_tracked_id_with_grasp = None
#                 if current_frame_grasps_to_visualize:  # Check again
#                     # Find a track_id associated with the grasps being visualized
#                     # This is a bit tricky if multiple objects are shown. Simplification:
#                     if current_frame_target_centroids:
#                         first_tracked_id_with_grasp = next(iter(current_frame_target_centroids))

#                 draw_image = grasp_planner.visualize_grasps(
#                     draw_image,
#                     current_frame_grasps_to_visualize,
#                     num_top_grasps=len(current_frame_grasps_to_visualize),
#                     # Show all found stable grasps for tracked targets
#                     object_centroid_abs=current_frame_target_centroids.get(
#                         first_tracked_id_with_grasp) if first_tracked_id_with_grasp else None
#                 )

#             frame_count += 1
#             elapsed_time = time.time() - start_time
#             if elapsed_time >= 1.0:
#                 fps = frame_count / elapsed_time
#                 frame_count, start_time = 0, time.time()
#             cv2.putText(draw_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             display_h = 720
#             scale_factor = display_h / draw_image.shape[0]
#             img_display = cv2.resize(draw_image, (int(draw_image.shape[1] * scale_factor), display_h))
#             cv2.imshow(f"Live Tracked {TARGET_CLASS_NAME} Grasping", img_display)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 print("Exit key 'q' pressed.")
#                 if not os.path.exists(output_folder): os.makedirs(output_folder)
#                 save_frame(color_image_bgr, output_folder, prefix="last_color")
#                 save_frame(draw_image, output_folder, prefix="last_detection_grasp_tracked")
#                 break
#     finally:
#         print("Stopping RealSense pipeline.")
#         if 'pipeline' in locals() and pipeline: pipeline.stop()
#         cv2.destroyAllWindows()
#         print("Script finished.")

# # if __name__ == "__main__":
# #     cam = CameraOperations()
# #     while True:
# #         try:
# #             x, y, z = cam.get_corners_translations()
# #             print(f"Marker location: x={x:.3f}, y={y:.3f}, z={z:.3f} m")
# #         except ValueError:
# #             print("Marker ID 0 not detected.")

# # if __name__ == "__main__":
# #     cam = CameraOperations()
# #     while True:
# #         try:
# #             transforms = cam.get_marker_transforms()
# #             for marker_id, T in transforms.items():
# #                 print(f"Marker {marker_id} transform:\n{T}")
# #             # x,y,z = cam.find_tennis()
# #         except ValueError:
# #             print("Marker ID 0 not detected.")

# # if __name__ == "__main__":
# #     cam = CameraOperations()
# #     while True:
# #         try:
# #             cam.find_aruco_codes_in_the_image()
# #             # print(f"Marker location: x={x:.3f}, y={y:.3f}, z={z:.3f} m")
# #             # x,y,z = cam.find_tennis()
# #         except ValueError:
# #             print("Some marker was not detected.")

# if __name__ == "__main__":
#     cam = CameraOperations()
#     #cam.show_rgb_and_depth()
#     cam.find_aruco_codes_in_the_image()
#     #cam.find_tennis()
