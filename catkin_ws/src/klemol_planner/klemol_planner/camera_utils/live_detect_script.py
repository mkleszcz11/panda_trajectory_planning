# --- START OF FILE live_detect_segment_kalman_grasp_tangent_stable_plot_v5_no_main_try.py ---

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import math
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any

# --- Kalman Filter Import ---
try:
    from kalman_filter import AsynchronousPredictiveKalmanFilter
    print("Imported AsynchronousPredictiveKalmanFilter from apkf_tracker.py")
except ImportError: print("Error: Could not import AsynchronousPredictiveKalmanFilter."); exit()

# --- RealSense Capture Functions Import ---
try:
    from capture_realsense_frame_yolo import setup_realsense_pipeline, get_aligned_frames, save_frame
    print("Imported capture functions from capture_realsense_frame_yolo.py")
except ImportError: print("Error: Could not import from capture_realsense_frame_yolo.py."); exit()

# ========================================
# === Configuration Constants ============
# ========================================
TARGET_CLASS_NAME: str = 'scissors'; CONFIDENCE_THRESHOLD: float = 0.5
CAM_WIDTH: int = 640; CAM_HEIGHT: int = 480; CAM_FPS: int = 30
PREDICTION_HORIZON: int = 15; KF_DT: float = 1.0 / CAM_FPS
KF_PROCESS_NOISE_STD: float = 0.02; KF_MEASUREMENT_NOISE_STDS: List[float] = [0.003, 0.003, 0.01]; KF_INITIAL_COV_DIAG: List[float] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
OUTPUT_FOLDER: str = "live_frames_seg_kalman_tangent_stable_grasp_plot"; DISPLAY_HEIGHT: int = 720
PLOT_BUFFER_SIZE: int = int(60 * CAM_FPS); PLOT_ALPHA: float = 0.2; PLOT_UPDATE_INTERVAL: float = 0.1
GRIPPER_MIN_WIDTH: float = 0.01; GRIPPER_MAX_WIDTH: float = 0.08
GRASP_MAX_SAMPLES: int = 500; GRASP_NORMAL_SMOOTHING: int = 5; GRASP_OPPOSING_ANGLE_THRES: float = 35
GRASP_MIN_CONTOUR_LEN: int = 20; GRASP_MIN_VALID_3D_CONTOUR_POINTS: int = 10

# ========================================
# === Utility Functions ==================
# ========================================

def convert_depth_to_phys_coord_using_realsense_intrinsics(
    x: int, y: int, depth: float, intrinsics: rs.intrinsics
) -> Optional[np.ndarray]:
    if intrinsics is None or depth <= 0: return None
    try:
        x_clamped = int(max(0, min(x, intrinsics.width - 1)))
        y_clamped = int(max(0, min(y, intrinsics.height - 1)))
        result_xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [x_clamped, y_clamped], depth)
        return np.array(result_xyz)
    except Exception as e: return None

def update_fill_between(
    axis: plt.Axes, fill_obj: Any, x_data: List[float], y1_data: np.ndarray, y2_data: np.ndarray, **kwargs
) -> Any:
    try:
        if fill_obj in axis.collections: fill_obj.remove()
        elif isinstance(fill_obj, list) and fill_obj and fill_obj[0] in axis.collections: axis.collections.remove(fill_obj[0])
    except Exception: pass
    if len(x_data) == len(y1_data) == len(y2_data) and len(x_data) > 0: return axis.fill_between(x_data, y1_data, y2_data, **kwargs)
    return fill_obj

# ========================================
# === Grasping Functions =================
# ========================================

def estimate_contour_normals_2d_vectorized(
    contour_2d: np.ndarray, smoothing_window: int = 5
) -> Optional[np.ndarray]:
    n_points = len(contour_2d)
    if n_points < 3: return None
    p_prev = np.roll(contour_2d, 1, axis=0); p_next = np.roll(contour_2d, -1, axis=0)
    tangents = p_next - p_prev
    norms = np.linalg.norm(tangents, axis=1); valid_mask = norms > 1e-6
    normals_2d = np.zeros_like(contour_2d, dtype=float)
    normals_2d[valid_mask, 0] = -tangents[valid_mask, 1] / norms[valid_mask]
    normals_2d[valid_mask, 1] = tangents[valid_mask, 0] / norms[valid_mask]
    invalid_indices = np.where(~valid_mask)[0]
    for idx in invalid_indices:
        prev_valid_idx = (idx - 1 + n_points) % n_points; search_count = 0
        while not valid_mask[prev_valid_idx] and search_count < n_points: prev_valid_idx = (prev_valid_idx - 1 + n_points) % n_points; search_count += 1
        if valid_mask[prev_valid_idx]: normals_2d[idx] = normals_2d[prev_valid_idx]
    if smoothing_window > 1 and n_points >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed_x = np.convolve(normals_2d[:, 0], kernel, mode='same'); smoothed_y = np.convolve(normals_2d[:, 1], kernel, mode='same')
        smoothed_normals = np.column_stack((smoothed_x, smoothed_y))
        smoothed_norms = np.linalg.norm(smoothed_normals, axis=1); valid_smooth_mask = smoothed_norms > 1e-6
        smoothed_normals[valid_smooth_mask] /= smoothed_norms[valid_smooth_mask, np.newaxis]
        smoothed_normals[~valid_smooth_mask] = normals_2d[~valid_smooth_mask]
        return smoothed_normals
    else: return normals_2d

def find_tangent_normal_grasp_stable(
    contour_2d: np.ndarray, normals_2d: np.ndarray, depth_frame: rs.depth_frame,
    intrinsics: rs.intrinsics, gripper_min_width: float, gripper_max_width: float,
    max_samples: int = 500, opposing_angle_threshold_deg: float = 35,
    min_depth: float = 0.1, max_depth: float = 5.0
) -> Optional[Tuple]:
    if contour_2d is None or normals_2d is None or len(contour_2d) < GRASP_MIN_CONTOUR_LEN: return None
    n_points = len(contour_2d); angle_dot_threshold: float = math.cos(math.radians(180.0 - opposing_angle_threshold_deg)); target_width: float = (gripper_min_width + gripper_max_width) / 2.0
    rng = np.random.default_rng(); valid_3d_points: Dict[int, np.ndarray] = {}
    for i, p2d in enumerate(contour_2d):
        try: depth: float = depth_frame.get_distance(p2d[0], p2d[1])
        except Exception: continue
        if min_depth < depth < max_depth:
            p3d: Optional[np.ndarray] = convert_depth_to_phys_coord_using_realsense_intrinsics(p2d[0], p2d[1], depth, intrinsics)
            if p3d is not None: valid_3d_points[i] = p3d
    if len(valid_3d_points) < GRASP_MIN_VALID_3D_CONTOUR_POINTS: return None
    valid_indices: List[int] = list(valid_3d_points.keys()); valid_candidates: List[Tuple[Tuple, float]] = []
    for _ in range(max_samples):
        if len(valid_indices) < 2: break
        idx1_orig, idx2_orig = rng.choice(valid_indices, 2, replace=False)
        p1_3d: np.ndarray = valid_3d_points[idx1_orig]; p2_3d: np.ndarray = valid_3d_points[idx2_orig]
        n1_2d: np.ndarray = normals_2d[idx1_orig]; n2_2d: np.ndarray = normals_2d[idx2_orig]
        norm_n1 = np.linalg.norm(n1_2d); norm_n2 = np.linalg.norm(n2_2d)
        if norm_n1 < 1e-6 or norm_n2 < 1e-6: continue
        dist_vec: np.ndarray = p2_3d - p1_3d; dist: float = np.linalg.norm(dist_vec)
        if not (gripper_min_width <= dist <= gripper_max_width): continue
        n1_dot_n2: float = np.dot(n1_2d, n2_2d)
        if n1_dot_n2 >= angle_dot_threshold: continue
        grasp_center: np.ndarray = (p1_3d + p2_3d) / 2.0; grasp_width: float = dist
        x_axis: np.ndarray = dist_vec / dist; z_axis: np.ndarray = np.array([0.0, 0.0, -1.0]); y_axis: np.ndarray = np.cross(z_axis, x_axis); y_norm: float = np.linalg.norm(y_axis)
        if y_norm < 1e-6: continue
        y_axis /= y_norm; z_axis = np.cross(x_axis, y_axis); grasp_orientation_matrix: np.ndarray = np.column_stack((x_axis, y_axis, z_axis))
        score: float = abs(grasp_width - target_width)
        grasp_tuple: Tuple = (grasp_center, grasp_orientation_matrix, grasp_width, p1_3d, p2_3d)
        valid_candidates.append((grasp_tuple, score))
    if not valid_candidates: return None
    valid_candidates.sort(key=lambda item: item[1]); best_grasp_tuple: Tuple = valid_candidates[0][0]
    return best_grasp_tuple

def get_grasp_candidate_from_mask(
    target_mask: Optional[np.ndarray], depth_frame: rs.depth_frame, intrinsics: rs.intrinsics
) -> Optional[Tuple]:
    if target_mask is None: return None
    contours_found, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_found: return None
    contour_2d: np.ndarray = max(contours_found, key=cv2.contourArea).squeeze()
    if contour_2d is None or contour_2d.ndim != 2 or len(contour_2d) <= GRASP_MIN_CONTOUR_LEN: return None
    normals_2d: Optional[np.ndarray] = estimate_contour_normals_2d_vectorized(contour_2d, smoothing_window=GRASP_NORMAL_SMOOTHING)
    if normals_2d is None: return None
    best_grasp: Optional[Tuple] = find_tangent_normal_grasp_stable(
        contour_2d, normals_2d, depth_frame, intrinsics,
        GRIPPER_MIN_WIDTH, GRIPPER_MAX_WIDTH,
        max_samples=GRASP_MAX_SAMPLES, opposing_angle_threshold_deg=GRASP_OPPOSING_ANGLE_THRES)
    return best_grasp

# ========================================
# === Visualization Functions ============
# ========================================

def draw_grasp_pose(
    image: np.ndarray, grasp_data: Optional[Tuple], intrinsics: rs.intrinsics, line_length: float = 0.05
) -> None:
    if grasp_data is None: return
    grasp_center, grasp_orientation, grasp_width, p_contact1_3d, p_contact2_3d = grasp_data
    if grasp_center is None or grasp_orientation is None or p_contact1_3d is None or p_contact2_3d is None: return
    try:
        center_uv: Tuple[int, int] = tuple(map(int, rs.rs2_project_point_to_pixel(intrinsics, grasp_center)))
        finger1_uv: Tuple[int, int] = tuple(map(int, rs.rs2_project_point_to_pixel(intrinsics, p_contact1_3d)))
        finger2_uv: Tuple[int, int] = tuple(map(int, rs.rs2_project_point_to_pixel(intrinsics, p_contact2_3d)))
        cv2.circle(image, finger1_uv, 6, (0, 255, 255), -1); cv2.drawMarker(image, finger1_uv, (0,0,0), cv2.MARKER_CROSS, 10, 1)
        cv2.circle(image, finger2_uv, 6, (0, 255, 255), -1); cv2.drawMarker(image, finger2_uv, (0,0,0), cv2.MARKER_CROSS, 10, 1)
        cv2.line(image, finger1_uv, finger2_uv, (0, 255, 0), 2)
        axis_1_end: np.ndarray = grasp_center + grasp_orientation[:, 0] * line_length; axis_1_uv: Tuple[int, int] = tuple(map(int, rs.rs2_project_point_to_pixel(intrinsics, axis_1_end)))
        axis_2_end: np.ndarray = grasp_center + grasp_orientation[:, 1] * line_length; axis_2_uv: Tuple[int, int] = tuple(map(int, rs.rs2_project_point_to_pixel(intrinsics, axis_2_end)))
        axis_3_end: np.ndarray = grasp_center + grasp_orientation[:, 2] * line_length; axis_3_uv: Tuple[int, int] = tuple(map(int, rs.rs2_project_point_to_pixel(intrinsics, axis_3_end)))
        cv2.line(image, tuple(center_uv), tuple(axis_1_uv), (0, 255, 0), 2); cv2.line(image, tuple(center_uv), tuple(axis_2_uv), (255, 0, 0), 2); cv2.line(image, tuple(center_uv), tuple(axis_3_uv), (0, 0, 255), 2)
    except Exception as e_vis: pass

def draw_kf_predictions(
    draw_image: np.ndarray, predicted_states_apriori: np.ndarray, intrinsics: rs.intrinsics
) -> None:
    max_pred_step = PREDICTION_HORIZON - 1; state_dim = 6
    for step in range(PREDICTION_HORIZON):
         current_pred_state = predicted_states_apriori[step*state_dim:(step+1)*state_dim]; pred_pos_3d = current_pred_state[0:3];
         print(f"predicted position 3D -> {pred_pos_3d}")
         if pred_pos_3d[2] > 0:
             try:
                 pred_u, pred_v = map(int,rs.rs2_project_point_to_pixel(intrinsics, pred_pos_3d));
                 if 0 <= pred_u < CAM_WIDTH and 0 <= pred_v < CAM_HEIGHT: ratio = step/max_pred_step if max_pred_step > 0 else 0; pred_color = (int(255*(1-ratio)), 0, int(255*ratio)); cv2.circle(draw_image, (pred_u, pred_v), 4, pred_color, -1);
             except Exception: pass

def setup_plotting() -> Dict:
    print(f"Initializing Plots (Buffer Size: {PLOT_BUFFER_SIZE} points)..."); plt.ion()
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    line_meas_x, = ax_x.plot([], [], 'rx', label='Measurement X', markersize=5); line_filt_x, = ax_x.plot([], [], 'b--', label='Filtered X (Posterior)'); fill_cov_x = ax_x.fill_between([], [], [], color='blue', alpha=PLOT_ALPHA, label='_nolegend_')
    ax_x.set_ylabel("X Position (m)"); ax_x.legend(loc='upper left'); ax_x.grid(True); ax_x.set_title(f"Kalman Filter Tracking: {TARGET_CLASS_NAME}")
    line_meas_y, = ax_y.plot([], [], 'gx', label='Measurement Y', markersize=5); line_filt_y, = ax_y.plot([], [], 'c--', label='Filtered Y (Posterior)'); fill_cov_y = ax_y.fill_between([], [], [], color='cyan', alpha=PLOT_ALPHA, label='_nolegend_')
    ax_y.set_ylabel("Y Position (m)"); ax_y.set_xlabel("Time (s)"); ax_y.legend(loc='upper left'); ax_y.grid(True)
    plot_handles = {'fig': fig, 'ax_x': ax_x, 'ax_y': ax_y, 'line_meas_x': line_meas_x, 'line_filt_x': line_filt_x, 'fill_cov_x': fill_cov_x, 'line_meas_y': line_meas_y, 'line_filt_y': line_filt_y, 'fill_cov_y': fill_cov_y}
    return plot_handles

def update_plot_data(
    plot_data: Dict, current_plot_time: float, measurement_coords: Tuple[float, float],
    filtered_coords: Tuple[float, float], std_devs: Tuple[float, float]
) -> None:
    plot_data['time'].append(current_plot_time)
    plot_data['meas_x'].append(measurement_coords[0])
    plot_data['meas_y'].append(measurement_coords[1])
    plot_data['filt_x'].append(filtered_coords[0])
    plot_data['filt_y'].append(filtered_coords[1])
    plot_data['std_x'].append(std_devs[0])
    plot_data['std_y'].append(std_devs[1])
    if len(plot_data['time']) > PLOT_BUFFER_SIZE:
        for key in plot_data: plot_data[key].pop(0)

def update_matplotlib_plots(plot_handles: Dict, plot_data: Dict) -> None:
    ph = plot_handles; pd = plot_data
    ph['line_meas_x'].set_data(pd['time'], pd['meas_x']); ph['line_filt_x'].set_data(pd['time'], pd['filt_x'])
    ph['fill_cov_x'] = update_fill_between(ph['ax_x'], ph['fill_cov_x'], pd['time'], np.array(pd['filt_x'])-np.array(pd['std_x']), np.array(pd['filt_x'])+np.array(pd['std_x']), color='blue', alpha=PLOT_ALPHA);
    ph['ax_x'].relim(); ph['ax_x'].autoscale_view()
    ph['line_meas_y'].set_data(pd['time'], pd['meas_y']); ph['line_filt_y'].set_data(pd['time'], pd['filt_y'])
    ph['fill_cov_y'] = update_fill_between(ph['ax_y'], ph['fill_cov_y'], pd['time'], np.array(pd['filt_y'])-np.array(pd['std_y']), np.array(pd['filt_y'])+np.array(pd['std_y']), color='cyan', alpha=PLOT_ALPHA);
    ph['ax_y'].relim(); ph['ax_y'].autoscale_view()
    try:
        if plt.fignum_exists(ph['fig'].number): ph['fig'].canvas.draw_idle(); ph['fig'].canvas.flush_events()
    except Exception as e_plot: print(f"Plot drawing error: {e_plot}")

# ========================================
# === Main Execution =====================
# ========================================

def main() -> None:
    """ Main execution function. """
    # --- Initialization outside loop ---
    pipeline = None
    fig = None
    model = None
    apkf = None

    # --- Setup Phase ---
    print("Setting up RealSense pipeline...")
    try:
        pipeline, profile, color_profile = setup_realsense_pipeline(request_max_res=False)
    except RuntimeError as e:
        print(f"FATAL: Failed RealSense init: {e}")
        return # Exit cleanly if setup fails

    if pipeline is None or profile is None or color_profile is None:
        print("FATAL: RealSense setup returned None.")
        return # Exit cleanly if setup fails

    color_intrinsics = color_profile.get_intrinsics()
    actual_format = color_profile.format()
    print(f"Pipeline started. Format: {actual_format}, Intrinsics: W={color_intrinsics.width}, H={color_intrinsics.height}")
    is_bgr_format = (actual_format == rs.format.bgr8)
    if not is_bgr_format:
        print("!!! WARNING: Non-BGR8 format detected. Colors might need conversion. !!!")

    if not torch.cuda.is_available():
        print("!!! WARNING: CUDA not available! Running on CPU. !!!")
        device = 'cpu'
    else:
        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        device = 'cuda'

    print(f"Loading YOLOv8 Segmentation model onto {device.upper()}...")
    try:
        model = YOLO('yolov8s-seg.pt')
        model.to(device)
        coco_names = model.names
        print(f"YOLOv8 model loaded successfully onto {device.upper()}.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}.") # Statement 1
        if 'pipeline' in locals() and pipeline:     # Statement 2
            pipeline.stop()                     # Statement 3 (conditional)
        exit()                                  # Statement 4

    print("Initializing Kalman Filter...")
    apkf = AsynchronousPredictiveKalmanFilter(N=PREDICTION_HORIZON, dt=KF_DT, process_noise_std=KF_PROCESS_NOISE_STD, initial_estimate_covariance_diag=KF_INITIAL_COV_DIAG)
    measurement_noise_vars = np.array(KF_MEASUREMENT_NOISE_STDS)**2
    apkf.R = np.diag(measurement_noise_vars)
    print(f"Set KF Measurement Noise R based on stds: {KF_MEASUREMENT_NOISE_STDS}")
    np.random.seed(42)
    mask_colors = np.random.randint(0, 256, (len(coco_names), 3), dtype=np.uint8)

    plot_handles = setup_plotting()
    fig = plot_handles['fig'] # Store for cleanup
    plot_data = {'time': [], 'meas_x': [], 'meas_y': [], 'filt_x': [], 'filt_y': [], 'std_x': [], 'std_y': []}
    occlusion_spans = []
    is_occluded = False
    occlusion_start_time = None

    print("\nStarting live detection loop... Press 'q' in CV window to quit.")
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()
    script_start_time = time.time()
    last_plot_update = script_start_time
    current_best_grasp: Optional[Tuple] = None

    # --- Main Loop ---
    while True:
        absolute_time: float = time.time()
        current_plot_time: float = absolute_time - script_start_time

        # --- Get Frames ---
        try:
            color_image_from_realsense, depth_frame = get_aligned_frames(pipeline, align_to=rs.stream.color)
        except RuntimeError as e:
            print(f"ERROR: RS capture error: {e}")
            time.sleep(0.5)
            continue
        if color_image_from_realsense is None or depth_frame is None:
            print("Warn: Skip frame (None received)")
            time.sleep(0.05)
            continue

        # --- Image Prep ---
        color_image_bgr: np.ndarray = color_image_from_realsense
        draw_image: np.ndarray = color_image_bgr.copy()
        overlay: np.ndarray = draw_image.copy()

        # --- YOLO Inference ---
        results = model(color_image_bgr, verbose=False)

        # --- Process Detections ---
        kf_measurement: Optional[np.ndarray] = None
        target_found_this_frame: bool = False
        measurement_coords_for_plot: Tuple[float, float] = (np.nan, np.nan)
        target_mask: Optional[np.ndarray] = None
        best_grasp_this_frame: Optional[Tuple] = None

        if results and results[0].boxes is not None and results[0].masks is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            masks_data = results[0].masks.data.cpu().numpy()
            current_img_shape_wh = (draw_image.shape[1], draw_image.shape[0])
            if masks_data.shape[1:] != (draw_image.shape[0], draw_image.shape[1]):
                masks_data = np.array([cv2.resize(m, current_img_shape_wh, interpolation=cv2.INTER_NEAREST) for m in masks_data])

            target_instance_found: bool = False
            for i in range(len(boxes)):
                conf: float = confs[i]
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, boxes[i])
                cls_id = int(clss[i])
                class_name = coco_names[cls_id]
                cx = (x1 + x2) // 2; cy = (y1 + y2) // 2; X, Y, Z_val = 0.0, 0.0, 0.0;
                try:
                    if 0 <= cx < CAM_WIDTH and 0 <= cy < CAM_HEIGHT:
                        depth_value = depth_frame.get_distance(cx, cy)
                        if 0.1 < depth_value < 5.0:
                             coords_3d = convert_depth_to_phys_coord_using_realsense_intrinsics(cx, cy, depth_value, color_intrinsics)
                             if coords_3d is not None:
                                 X, Y, Z_val = coords_3d
                             else:
                                 Z_val = 0.0
                        else:
                            Z_val = 0.0
                except Exception:
                    Z_val = 0.0

                if not target_instance_found and class_name == TARGET_CLASS_NAME and Z_val > 0:
                    kf_measurement = np.array([X, Y, Z_val])
                    target_found_this_frame = True
                    measurement_coords_for_plot = (X, Y)
                    if i < len(masks_data):
                        target_mask = masks_data[i].astype(np.uint8)
                    target_instance_found = True

                cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                try:
                    if i < len(masks_data):
                        overlay[masks_data[i].astype(bool)] = mask_colors[cls_id].tolist()
                except Exception as e_mask:
                    print(f"Mask draw error: {e_mask}")
                label = f"{class_name} {conf:.2f}"
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(draw_image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Find Grasp Candidate ---
        if target_mask is not None:
            best_grasp_this_frame = get_grasp_candidate_from_mask(target_mask, depth_frame, color_intrinsics)

        # --- Update Stored Grasp ---
        if best_grasp_this_frame is not None:
            current_best_grasp = best_grasp_this_frame
        elif not target_found_this_frame:
            current_best_grasp = None
        # --- End Detection Processing ---

        # --- Update KF ---
        predicted_states_apriori, _ = apkf.update(kf_measurement)
        current_posterior_state, current_posterior_cov = apkf.get_current_estimate()
        filtered_coords_for_plot = (current_posterior_state[0], current_posterior_state[1])
        epsilon = 1e-9
        current_std_x = np.sqrt(max(epsilon, current_posterior_cov[0, 0]))
        current_std_y = np.sqrt(max(epsilon, current_posterior_cov[1, 1]))
        std_devs_for_plot = (current_std_x, current_std_y)

        # --- Update Plotting Data & Logic ---
        update_plot_data(plot_data, current_plot_time, measurement_coords_for_plot, filtered_coords_for_plot, std_devs_for_plot)
        if not target_found_this_frame:
            if not is_occluded:
                is_occluded = True
                occlusion_start_time = current_plot_time
        else:
            if is_occluded:
                if occlusion_start_time is not None and occlusion_start_time < current_plot_time:
                    occlusion_spans.append((occlusion_start_time, current_plot_time))
                    plot_handles['ax_x'].axvspan(occlusion_start_time, current_plot_time, color='grey', alpha=0.3, zorder=-1)
                    plot_handles['ax_y'].axvspan(occlusion_start_time, current_plot_time, color='grey', alpha=0.3, zorder=-1)
                is_occluded = False
                occlusion_start_time = None
        if absolute_time - last_plot_update > PLOT_UPDATE_INTERVAL:
            update_matplotlib_plots(plot_handles, plot_data)
            last_plot_update = absolute_time

        # --- Visualize Frame ---
        draw_kf_predictions(draw_image, predicted_states_apriori, color_intrinsics)
        draw_grasp_pose(draw_image, current_best_grasp, color_intrinsics) # Visualize grasp

        # Combine Overlay, Add FPS, Display
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)
        frame_count += 1
        current_time_fps = time.time()
        elapsed_time_fps = current_time_fps - fps_start_time
        if elapsed_time_fps >= 1.0:
            fps = frame_count / elapsed_time_fps
            frame_count = 0
            fps_start_time = current_time_fps
        cv2.putText(draw_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if draw_image.shape[0] != DISPLAY_HEIGHT:
            scale = DISPLAY_HEIGHT / draw_image.shape[0]
            img_display = cv2.resize(draw_image, (int(draw_image.shape[1]*scale), DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
        else:
            img_display = draw_image
        cv2.imshow("Live Segmentation + Kalman + Stable Tangent Grasp", img_display)

        # --- Exit Condition ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exit key pressed.")
            break
    # --- End While Loop ---


# ========================================
# === Script Entry Point ===============
# ========================================
if __name__ == "__main__":
    main()

# --- END OF FILE live_detect_segment_kalman_grasp_tangent_stable_plot_v5_fixed.py ---