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
from klemol_planner.goals.point_with_orientation import PointWithOrientation

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
        Initialize RealSense pipeline, YOLO model, and grasp planner.
        """
        self.USE_REALSENSE = True
        self.pipeline = None

        if self.USE_REALSENSE:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

            self.pipeline.start(config)
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            profile = self.pipeline.get_active_profile()
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            self.color_width = intr.width
            self.color_height = intr.height
            self.color_intrinsics = intr

            self.camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                                           [0, intr.fy, intr.ppy],
                                           [0, 0, 1]])
            self.dist_coeffs = np.array(intr.coeffs)

        else:
            self._use_default_intrinsics()

        # ArUco settings
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.marker_length = 0.05  # meters

        # Device and model setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"YOLO running on: {self.device.upper()}")
        self.model = YOLO("yolov8x-seg.pt").to(self.device)
        # self.model = YOLO("https://universe.roboflow.com/roboflow-100/yolov8x-roboflow100/1")
        # self.class_names = ["fork", "spoon", "banana", "scissors"] 
        self.coco_names = self.model.names

        # # Grasp planner parameters
        # self.grasp_planner = AntipodalGraspPlanner(
        #     max_gripper_opening_px=120,
        #     min_grasp_width_px=15,
        #     angle_tolerance_deg=20,
        #     contour_approx_epsilon_factor=0.001,
        #     normal_neighborhood_k=30,
        #     dist_penalty_weight=0.01,
        #     width_favor_narrow_weight=0.01
        # )

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
        # cv2.imshow("ArUco Detection", cv2.resize(color_image, (1280, 720)))
        # cv2.waitKey(0)

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


    def get_list_of_picking_points_in_camera_frame(self, objects_names: t.List[str] = None, timeout: float = 10.0, points_to_not_focus_on: t.List[PointWithOrientation] = None) -> t.List[t.Tuple[str, PointWithOrientation]]:
        """
        #TODO - THIS FUNCTION SHOULD BE CHANGED, IT DOES NOT ACCOUNT FOR MULTIPLE INSTANCES OF THE SAME OBJECT
        Get a list of picking points for the specified objects.

        Args:
            objects_names (t.List[str], optional): List of object names to find picking points for.
            timeout (float, optional): Timeout for the operation.
            points_to_not_focus_on (t.List[PointWithOrientation], optional): List of points to ignore when finding picking points.

        Returns:
            t.List[t.Tuple[str, PointWithOrientation]]: List of tuples containing object name and picking point.
        """
        if objects_names is None:
            objects_names = self.object_name_to_aruco.keys()

        picking_points = []
        for object_name in objects_names:
            success, x, y, z, yaw = self.find_grasp(object_name, timeout=timeout, points_to_not_focus_on=points_to_not_focus_on, number_of_tries=5)
            if success:
                yaw = yaw % math.pi
                point_in_camera_frame = PointWithOrientation(
                    x = x,
                    y = y,
                    z = z,
                    roll = 0.0,
                    pitch = 0.0,
                    yaw = (-math.pi * 0.25) + yaw
                )
                if point_in_camera_frame:
                    picking_points.append((object_name, point_in_camera_frame))
            else:
                continue

        return picking_points
    
    def get_list_of_bb_centers_for_picking_points_in_camera_frame(self, objects_names: t.List[str] = None, timeout: float = 10.0, points_to_not_focus_on: t.List[PointWithOrientation] = None) -> t.List[t.Tuple[str, PointWithOrientation]]:
        """
        Same as get_list_of_picking_points_in_camera_frame(), but the return is the xy of the bounding box.
        """
        if objects_names is None:
            objects_names = self.object_name_to_aruco.keys()

        picking_points = []
        for object_name in objects_names:
            success, x, y, z, yaw = self.find_bb(object_name, timeout=timeout, points_to_not_focus_on=points_to_not_focus_on, number_of_tries=5)
            if success:
                yaw = yaw % math.pi
                point_in_camera_frame = PointWithOrientation(
                    x = x,
                    y = y,
                    z = z,
                    roll = 0.0,
                    pitch = 0.0,
                    yaw = (-math.pi * 0.25) + yaw
                )
                if point_in_camera_frame:
                    picking_points.append((object_name, point_in_camera_frame))
            else:
                continue

        return picking_points


    def find_bb(self, object_to_find="sports ball", timeout=10.0,
                points_to_not_focus_on=None, number_of_tries=None) -> t.List:
        """
        Find the bounding box center (in 3D) of a specified object class.
        Returns (success, x, y, z, yaw).
        """
        if self.pipeline is None:
            return False, None, None, None, None

        confidence_threshold = 0.3
        current_try = 0
        start_time = time.time()

        while True:
            if number_of_tries and current_try >= number_of_tries:
                break
            if time.time() - start_time > timeout:
                break

            current_try += 1
            color_img, depth_frame = get_aligned_frames(self.pipeline, align_to=rs.stream.color)
            if color_img is None or depth_frame is None:
                continue

            if points_to_not_focus_on:
                masked_img = color_img.copy()
                radius = 30
                for pt in points_to_not_focus_on:
                    x, y = int(pt.x), int(pt.y)
                    cv2.circle(masked_img, (x, y), radius, (0, 0, 0), thickness=-1)
                color_img = masked_img

            # YOLO expects RGB
            rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            results = self.model.track(rgb_img, persist=True, tracker="botsort.yaml", verbose=False)
            if not results or results[0].boxes is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [-1] * len(boxes)

            draw_img = color_img.copy()

            for box, conf, cls_id, track_id in zip(boxes, confs, clss, track_ids):
                if conf < confidence_threshold:
                    continue
                if self.coco_names[cls_id] != object_to_find or track_id == -1:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if points_to_not_focus_on:
                    skip = False
                    for pt in points_to_not_focus_on:
                        dist = math.hypot(pt.x - cx, pt.y - cy)
                        if dist < radius:
                            skip = True
                            break
                    if skip:
                        continue

                # Draw bounding box and label
                label = f"{self.coco_names[cls_id]} {conf:.2f}"
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(draw_img, label, (x1, y1 - 10 if y1 > 20 else y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw center point
                cv2.circle(draw_img, (cx, cy), 5, (255, 0, 0), -1)

                # Get depth and return 3D point
                depth = depth_frame.get_distance(cx, cy)
                if not (0.1 < depth < 5.0):
                    continue

                coords_3d = convert_depth_to_phys_coord_using_realsense_intrinsics(cx, cy, depth, self.color_intrinsics)
                if coords_3d is None:
                    continue

                x, y, z = coords_3d

                # Show live display (non-blocking)
                disp_h = 720
                scale = disp_h / draw_img.shape[0]
                display_img = cv2.resize(draw_img, (int(draw_img.shape[1] * scale), disp_h))
                # cv2.imshow("YOLO Bounding Boxes", display_img)
                # cv2.waitKey(1)

                return True, x, y, z, 0.0

            # Show frame even if no box matched yet
            disp_h = 720
            scale = disp_h / draw_img.shape[0]
            display_img = cv2.resize(draw_img, (int(draw_img.shape[1] * scale), disp_h))
            cv2.imshow("YOLO Bounding Boxes", display_img)
            cv2.waitKey(1)

        cv2.destroyWindow("YOLO Bounding Boxes")
        return False, None, None, None, None

    # def find_grasp_v2(self, object_to_find="sports ball", timeout=10.0,
    #             points_to_not_focus_on=None, number_of_tries=None) -> t.List:
    #     """
    #     Find a grasp for a specified object class. Returns (success, x, y, z, yaw)
    #     """

    # THIS IS A NEW IMPLMENTATION - OLD ONE IS COMMENTED OUT BELOW 
    def find_grasp(self, object_to_find="sports ball", timeout=10.0,
                points_to_not_focus_on=None, number_of_tries=None) -> t.List:
        """
        NEW IMPLEMENTATION
        Detects the specified object and returns the grasp point based on the oriented bounding box.

        Returns:
            Tuple[bool, float, float, float, float]: success, x, y, z, yaw
        """
        if self.pipeline is None:
            return False, None, None, None, None

        confidence_threshold = 0.4
        start_time = time.time()

        while time.time() - start_time < timeout:
            color_img, depth_frame = get_aligned_frames(self.pipeline, align_to=rs.stream.color)
            if color_img is None or depth_frame is None:
                continue

            results = self.model.track(color_img, persist=True, tracker="botsort.yaml", verbose=False)
            if not results or results[0].boxes is None or results[0].masks is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy()
            shape = color_img.shape[:2]

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, clss)):
                if conf < confidence_threshold or self.coco_names[cls_id] != object_to_find:
                    continue

                mask = cv2.resize(masks[i], (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask > 0.5).astype(np.uint8) * 255

                coords = np.column_stack(np.where(binary_mask > 0))
                if coords.shape[0] < 5:
                    continue

                # Fit rotated bounding box
                rot_rect = cv2.minAreaRect(coords)
                (cx, cy), (w, h), angle = rot_rect

                if w < h:
                    angle += 90
                yaw = np.deg2rad(angle)

                # Get depth
                depth = depth_frame.get_distance(int(cx), int(cy))
                if depth == 0:
                    continue

                x, y, z = convert_depth_to_phys_coord_using_realsense_intrinsics(int(cx), int(cy), depth, self.color_intrinsics)
                return True, x, y, z, yaw

        return False, None, None, None, None

    # def find_grasp(self, object_to_find="sports ball", timeout=10.0,
    #             points_to_not_focus_on=None, number_of_tries=None) -> t.List:
    #     """
    #     Find a grasp for a specified object class. Returns (success, x, y, z, yaw)
    #     """
    #     if self.pipeline is None:
    #         return False, None, None, None, None

    #     confidence_threshold = 0.4
    #     output_folder = f"live_{object_to_find.replace(' ', '_')}_grasps_tracked"
    #     mask_colors = np.random.randint(0, 256, (len(self.coco_names), 3), dtype=np.uint8)

    #     tracked_best_grasps = {}
    #     is_grasp_found = False
    #     current_try = 0
    #     start_time = time.time()

    #     while not is_grasp_found:
    #         if number_of_tries and current_try >= number_of_tries:
    #             break

    #         if time.time() - start_time > timeout:
    #             break

    #         current_try += 1
    #         color_img, depth_frame = get_aligned_frames(self.pipeline, align_to=rs.stream.color)
    #         if color_img is None or depth_frame is None:
    #             continue

    #         # Optional mask visualisation with ignore points
    #         if points_to_not_focus_on:
    #             draw_img = color_img.copy()
    #             masked_img = color_img.copy()
    #             radius = 30

    #             for pt in points_to_not_focus_on:
    #                 x, y = int(pt.x), int(pt.y)
    #                 cv2.circle(masked_img, (x, y), radius, (0, 0, 0), thickness=-1)
    #                 cv2.circle(draw_img, (x, y), radius, (0, 0, 255), thickness=2)
    #                 cv2.putText(draw_img, "Ignore", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #             # Show masked and annotated image
    #             disp_h = 720
    #             scale = disp_h / masked_img.shape[0]
    #             vis_img = cv2.resize(draw_img, (int(draw_img.shape[1] * scale), disp_h))
    #             cv2.imshow("Ignore Points Visualisation", vis_img)
    #             cv2.waitKey(1)
    #             # cv2.destroyWindow("Ignore Points Visualisation")

    #             # Apply mask to input for detection
    #             color_img = masked_img

    #         results = self.model.track(color_img, persist=True, tracker="botsort.yaml", verbose=False)
    #         if not results or results[0].boxes is None or results[0].masks is None:
    #             continue

    #         boxes = results[0].boxes.xyxy.cpu().numpy()
    #         confs = results[0].boxes.conf.cpu().numpy()
    #         clss = results[0].boxes.cls.cpu().numpy()
    #         track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [-1] * len(boxes)
    #         masks = results[0].masks.data.cpu().numpy()
    #         shape = color_img.shape[:2]

    #         for i, (box, conf, cls_id, track_id) in enumerate(zip(boxes, confs, clss, track_ids)):
    #             if conf < confidence_threshold or self.coco_names[cls_id] != object_to_find or track_id == -1:
    #                 continue

    #             mask = cv2.resize(masks[i], (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    #             binary_mask = (mask > 0.5).astype(np.uint8) * 255

    #             # Add after: if conf < threshold ... etc.
    #             obj_cx = int((box[0] + box[2]) / 2)
    #             obj_cy = int((box[1] + box[3]) / 2)

    #             if points_to_not_focus_on:
    #                 for pt in points_to_not_focus_on:
    #                     dist = math.hypot(pt.x - obj_cx, pt.y - obj_cy)
    #                     if dist < radius:
    #                         continue  # skip this detection entirely

    #             new_grasps, centroid = self.grasp_planner.find_grasps(binary_mask.copy())

    #             if not new_grasps or centroid is None:
    #                 continue

    #             best_grasp = new_grasps[0]
    #             if track_id not in tracked_best_grasps or best_grasp['score'] > tracked_best_grasps[track_id]['score']:
    #                 tracked_best_grasps[track_id] = best_grasp

    #             abs_grasp = self.grasp_planner.transform_grasp_to_image_space(best_grasp, centroid)
    #             if not abs_grasp:
    #                 continue

    #             # Convert grasp to 3D
    #             p1, p2, center = abs_grasp['p1'], abs_grasp['p2'], abs_grasp['center_px']
    #             d1 = depth_frame.get_distance(*p1)
    #             d2 = depth_frame.get_distance(*p2)
    #             dc = depth_frame.get_distance(*center)

    #             p1_3d = convert_depth_to_phys_coord_using_realsense_intrinsics(p1[0], p1[1], d1, self.color_intrinsics)
    #             p2_3d = convert_depth_to_phys_coord_using_realsense_intrinsics(p2[0], p2[1], d2, self.color_intrinsics)

    #             if not all(p1_3d) or not all(p2_3d):
    #                 continue

    #             x = (p1_3d[0] + p2_3d[0]) / 2
    #             y = (p1_3d[1] + p2_3d[1]) / 2
    #             z = dc
    #             # Compute yaw from the rotated bounding box (minAreaRect)
    #             coords = np.column_stack(np.where(binary_mask > 0))
    #             if coords.shape[0] < 5:
    #                 continue  # Need at least 5 points to compute minAreaRect

    #             rot_rect = cv2.minAreaRect(coords)
    #             angle = rot_rect[2]

    #             # OpenCV returns angle in range [-90, 0), with respect to horizontal
    #             if rot_rect[1][0] < rot_rect[1][1]:  # width < height
    #                 angle = angle + 90

    #             yaw = np.deg2rad(angle)

    #             ###############################################
    #             # SHOW GRASP
    #             # --- Visualise grasp on the original image ---
    #             img_vis = color_img.copy()

    #             # Draw origin
    #             origin = (0, 0)
    #             x_axis_end = (origin[0] + 80, origin[1])
    #             y_axis_end = (origin[0], origin[1] + 80)

    #             cv2.arrowedLine(img_vis, origin, x_axis_end, (0, 0, 255), 2, tipLength=0.2)  # X in red
    #             cv2.arrowedLine(img_vis, origin, y_axis_end, (0, 255, 0), 2, tipLength=0.2)  # Y in green
    #             cv2.putText(img_vis, "Origin (0,0)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    #             cv2.putText(img_vis, "X", (x_axis_end[0] + 5, x_axis_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #             cv2.putText(img_vis, "Y", (y_axis_end[0] + 5, y_axis_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #             # Draw grasp points
    #             cv2.circle(img_vis, p1, 5, (0, 255, 0), -1)  # P1 green
    #             cv2.circle(img_vis, p2, 5, (0, 255, 0), -1)  # P2 green
    #             cv2.circle(img_vis, center, 5, (255, 0, 0), -1)  # center blue

    #             cv2.putText(img_vis, "P1", (p1[0] + 5, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #             cv2.putText(img_vis, "P2", (p2[0] + 5, p2[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #             # Resize for display
    #             display_h = 720
    #             scale_factor = display_h / img_vis.shape[0]
    #             img_resized = cv2.resize(img_vis, (int(img_vis.shape[1] * scale_factor), display_h))
    #             cv2.imshow("Detected Grasp", img_resized)
    #             cv2.waitKey(1)
    #             cv2.destroyWindow("Detected Grasp")
    #             #####################################
                
    #             return True, x, y, z, yaw

    #     return False, None, None, None, None


def main():
    cam = CameraOperations()

    print("Press 'g' to detect grasp candidates.")
    print("Press 'b' to detect bounding box centers.")
    print("Press 'a' to detect and print ArUco markers.")
    print("Press 'i' to show RGB + Depth window.")
    print("Press 'q' to quit.")

    while True:
        key = input("Enter command: ").strip().lower()

        if key == 'g':
            print("Finding grasp points (WIP)...")
            # Replace with working version when ready
            results = cam.get_list_of_picking_points_in_camera_frame(objects_names=["fork", "spoon", "banana", "scissors"])
            for obj, point in results:
                print(f"[GRASP] {obj}: {point}")
        elif key == 'b':
            print("Finding bounding box centers...")
            results = cam.get_list_of_bb_centers_for_picking_points_in_camera_frame(objects_names=["fork", "spoon", "banana", "scissors"])
            for obj, point in results:
                print(f"[BB] {obj}: {point}")
        elif key == 'a':
            print("Detecting ArUco markers...")
            cam.get_marker_transforms()
        elif key == 'i':
            cam.show_rgb_and_depth()
        elif key == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid command. Try again.")


if __name__ == "__main__":
    main()
