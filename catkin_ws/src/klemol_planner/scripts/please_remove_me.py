# --- START OF FILE live_detect_segment_kalman_plot.py ---

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import math
import torch
import matplotlib.pyplot as plt  # Import Matplotlib

# Grasp Candidate Imports
from scipy.spatial import ConvexHull
import scipy.ndimage as ndimage

# # --- Kalman Filter Import ---
# try:
#     # Ensure this file exists and contains the class definition
#     from apkf_tracker import AsynchronousPredictiveKalmanFilter

#     print("Imported AsynchronousPredictiveKalmanFilter from apkf_tracker.py")
# except ImportError:
#     print("Error: Could not import AsynchronousPredictiveKalmanFilter.")
# #     print("Please ensure apkf_tracker.py exists or paste the class definition here.")
# #     exit()

# # --- RealSense Capture Functions Import ---
# try:
#     # Ensure this file exists and contains the necessary functions
#     from capture_realsense_frame_yolo import setup_realsense_pipeline, get_aligned_frames, save_frame

#     print("Imported capture functions from capture_realsense_frame_yolo.py")
# except ImportError:
#     print("Error: Could not import from capture_realsense_frame_yolo.py.")
#     exit()


# --- Grasp Candidate Generation ---
class GraspCandidateGenerator:
    def __init__(self, max_grasp_distance=0.1, angle_threshold=30):
        """
        Initialize Grasp Candidate Generator
        :param max_grasp_distance: Maximum distance between grasp points (meters)
        :param angle_threshold: Maximum angle deviation from antipodal orientation (degrees)
        """
        self.max_grasp_distance = max_grasp_distance
        self.angle_threshold = angle_threshold
        self.grasp_candidates = []

    def estimate_surface_normal(self, mask):
        """
        Estimate surface normal using image gradients
        :param mask: Binary mask of the object
        :return: Gradient magnitude and direction
        """
        # Compute image gradients
        dx = ndimage.sobel(mask, axis=0)
        dy = ndimage.sobel(mask, axis=1)

        # Compute gradient magnitude and direction
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        direction = np.arctan2(dy, dx)

        return magnitude, direction

    def generate_grasp_candidates(self, mask, depth_frame=None, intrinsics=None):
        """
        Generate grasp candidates from a binary mask
        :param mask: Binary segmentation mask
        :param depth_frame: Optional depth frame for 3D coordinate conversion
        :param intrinsics: Camera intrinsics for depth-to-3D conversion
        :return: List of grasp candidates [(point1, point2, score), ...]
        """
        # Reset candidates for this frame
        self.grasp_candidates = []

        # Check if mask is valid
        if mask is None or np.sum(mask) == 0:
            return []

        # Estimate surface normals
        mag, ang = self.estimate_surface_normal(mask)

        # Find potential grasp points (high gradient magnitude regions)
        candidate_points = np.column_stack(np.where(mag > np.percentile(mag, 75)))

        # Look for antipodal point pairs
        for i, p1 in enumerate(candidate_points):
            for p2 in candidate_points[i + 1:]:
                # Compute inter-point distance and angle between normals
                dist = np.linalg.norm(p1 - p2)

                # Check distance constraint
                if dist > self.max_grasp_distance * (mask.shape[0] / intrinsics.height):
                    continue

                # Compute angle between surface normals
                normal1 = ang[p1[0], p1[1]]
                normal2 = ang[p2[0], p2[1]]
                angle_diff = abs(np.degrees(normal1 - normal2 + np.pi))

                # Check antipodal condition
                if angle_diff < self.angle_threshold or abs(angle_diff - 180) < self.angle_threshold:
                    # Optional: Convert to 3D coordinates if depth available
                    if depth_frame is not None and intrinsics is not None:
                        try:
                            # Convert 2D points to 3D
                            x1, y1 = p1[1], p1[0]  # OpenCV uses (x,y) order
                            x2, y2 = p2[1], p2[0]
                            z1 = depth_frame.get_distance(x1, y1)
                            z2 = depth_frame.get_distance(x2, y2)

                            # Only add if both points have valid depth
                            if z1 > 0 and z2 > 0:
                                # Convert to 3D coordinates using RealSense intrinsics
                                point3d1 = rs.rs2_deproject_pixel_to_point(intrinsics, [x1, y1], z1)
                                point3d2 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2, y2], z2)

                                # Compute grasp score (antipodality + depth consistency)
                                grasp_score = 1.0 / (angle_diff + 1) * abs(z1 - z2)

                                self.grasp_candidates.append((point3d1, point3d2, grasp_score))
                        except Exception as e:
                            print(f"Depth conversion error: {e}")

        # Sort candidates by score (higher is better)
        self.grasp_candidates.sort(key=lambda x: x[2], reverse=True)

        return self.grasp_candidates


# --- Depth Conversion Function ---
def convert_depth_to_phys_coord_using_realsense_intrinsics(x, y, depth, intrinsics):
    # Converts pixel coordinates (u, v) and depth to 3D coordinates (X, Y, Z)
    if intrinsics is None or depth <= 0: return 0.0, 0.0, 0.0
    try:
        # Clamp coordinates to be within the frame dimensions
        x = int(max(0, min(x, intrinsics.width - 1)))
        y = int(max(0, min(y, intrinsics.height - 1)))
        # Deproject pixel to point using RealSense SDK
        result = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        # result is [X, Y, Z] in meters
        return result[0], result[1], result[2]
    except Exception:
        # Return origin or handle error if deprojection fails
        return 0.0, 0.0, 0.0


# --- Robust Fill Between Update ---
def update_fill_between(axis, fill_obj, x_data, y1_data, y2_data, **kwargs):
    # Helper function to reliably remove and redraw fill_between objects in Matplotlib
    try:
        # Attempt to remove the old fill object
        if fill_obj in axis.collections:
            fill_obj.remove()
        # Handle case where fill_obj might be a list (older matplotlib)
        elif isinstance(fill_obj, list) and fill_obj and fill_obj[0] in axis.collections:
            axis.collections.remove(fill_obj[0])
    except Exception:
        pass  # Ignore if removal fails (e.g., object already removed)

    # Redraw the fill_between if data is valid
    if len(x_data) == len(y1_data) == len(y2_data) and len(x_data) > 0:
        return axis.fill_between(x_data, y1_data, y2_data, **kwargs)
    # Return the old (or potentially removed) object handle if data is invalid
    return fill_obj


# --- Configuration ---
# YOLO / Detection
TARGET_CLASS_NAME = 'banana'  # Object class to track with Kalman Filter
CONFIDENCE_THRESHOLD = 0.5  # Minimum detection confidence
# RealSense (matches setup_realsense_pipeline with request_max_res=False)
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30  # Default resolution and FPS
# Kalman Filter (Tune these carefully!)
PREDICTION_HORIZON = 15  # Predict N steps ahead (e.g., 0.5 seconds at 30 FPS)
KF_DT = 1.0 / CAM_FPS  # Use camera FPS for filter timestep
KF_PROCESS_NOISE_STD = 0.1  # Process noise (acceleration uncertainty m/s^2) - *** TUNE ME ***
# Use experimentally derived measurement noise (or estimate if unknown)
KF_MEASUREMENT_NOISE_STDS = [0.005, 0.005,
                             0.01]  # Measurement noise [X, Y, Z] meters - *** TUNE ME *** (Increased from tennis ball)
KF_INITIAL_COV_DIAG = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5]  # Initial uncertainty [Pos_XYZ, Vel_XYZ] - *** TUNE ME ***
# Output / Display
OUTPUT_FOLDER = "live_frames_seg_kalman_plot"  # Folder for saving images
DISPLAY_HEIGHT = 720  # Target height for the OpenCV display window

# Plotting Config
# Store enough points for ~60 seconds of history (Points = Seconds * FPS)
PLOT_BUFFER_SIZE = 60 * CAM_FPS  # Calculate based on desired history and FPS
PLOT_ALPHA = 0.2  # Transparency for covariance fill
PLOT_UPDATE_INTERVAL = 0.1  # Update plot graphics every X seconds

# --- Main Script Logic ---

# # 1. Setup RealSense Pipeline
# print("Setting up RealSense pipeline...")
# pipeline, profile, color_profile = setup_realsense_pipeline(request_max_res=False)
# if pipeline is None: print("Failed RealSense init."); exit()
# color_intrinsics = color_profile.get_intrinsics()
# actual_format = color_profile.format()
# print(f"Pipeline started. Format: {actual_format}, Intrinsics: W={color_intrinsics.width}, H={color_intrinsics.height}")
# is_bgr_format = (actual_format == rs.format.bgr8)
# if not is_bgr_format: print("!!! WARNING: Non-BGR8 format detected. Colors might need conversion. !!!")

# 2. Check for CUDA GPU
if not torch.cuda.is_available():
    print("!!! WARNING: CUDA not available! Running on CPU. !!!")
    device = 'cpu'
else:
    print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    device = 'cuda'

# 3. Load YOLO Model
print(f"Loading YOLO Segmentation model onto {device.upper()}...")
model = None
try:
    model = YOLO('yolo11x-seg.pt')
    model.to(device)
    coco_names = model.names
    print(f"YOLO model loaded successfully onto {device.upper()}.")
except Exception as e:
    print(f"Error loading YOLO model: {e}.")
    if 'pipeline' in locals() and pipeline: pipeline.stop()
    exit()

# 4. Initialize Kalman Filter
print("Initializing Kalman Filter...")
apkf = AsynchronousPredictiveKalmanFilter(N=PREDICTION_HORIZON, dt=KF_DT, process_noise_std=KF_PROCESS_NOISE_STD,
                                          initial_estimate_covariance_diag=KF_INITIAL_COV_DIAG)
measurement_noise_vars = np.array(KF_MEASUREMENT_NOISE_STDS) ** 2;
apkf.R = np.diag(measurement_noise_vars)
print(f"Set KF Measurement Noise R based on stds: {KF_MEASUREMENT_NOISE_STDS}")

# Colors for masks
np.random.seed(42);
mask_colors = np.random.randint(0, 256, (len(coco_names), 3), dtype=np.uint8)

# Initialize Grasp Candidate Generator
grasp_generator = GraspCandidateGenerator()

# --- Plot Initialization ---
print("Initializing Plots...")
plt.ion()  # Turn on interactive mode for Matplotlib
fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# Plot X data
line_meas_x, = ax_x.plot([], [], 'rx', label='Measurement X', markersize=5)
line_filt_x, = ax_x.plot([], [], 'b--', label='Filtered X (Posterior)')
fill_cov_x = ax_x.fill_between([], [], [], color='blue', alpha=PLOT_ALPHA, label='_nolegend_')  # Dummy fill
ax_x.set_ylabel("X Position (m)");
ax_x.legend(loc='upper left');
ax_x.grid(True)
ax_x.set_title(f"Kalman Filter Tracking: {TARGET_CLASS_NAME}")
# Plot Y data
line_meas_y, = ax_y.plot([], [], 'gx', label='Measurement Y', markersize=5)
line_filt_y, = ax_y.plot([], [], 'c--', label='Filtered Y (Posterior)')
fill_cov_y = ax_y.fill_between([], [], [], color='cyan', alpha=PLOT_ALPHA, label='_nolegend_')
ax_y.set_ylabel("Y Position (m)");
ax_y.set_xlabel("Time (s)");
ax_y.legend(loc='upper left')
ax_y.grid(True)
# Data storage for plots
plot_time = [];
plot_meas_x, plot_meas_y = [], []
plot_filt_x, plot_filt_y = [], []
plot_std_x, plot_std_y = [], []
occlusion_spans = [];
is_occluded = False
occlusion_start_time = None
# -------------------------

print("\nStarting live detection loop... Press 'q' in CV window to quit.")
# --- Initialize FPS counter variables ---
fps = 0.0
frame_count = 0
fps_start_time = time.time()  # <<< CORRECTED: Dedicated start time for FPS calculation
script_start_time = time.time()  # <<< Start time for plot X-axis
last_plot_update = script_start_time
# ---------------------------------------

try:
    while True:
        absolute_time = time.time()
        # Use time relative to the script start for the plot's X-axis
        current_plot_time = absolute_time - script_start_time

        # 5. Capture Aligned Frames
        color_image_from_realsense, depth_frame = get_aligned_frames(pipeline, align_to=rs.stream.color)
        if color_image_from_realsense is None or depth_frame is None: time.sleep(0.01); continue

        # Assume input is BGR (check 'actual_format' printout if colors are wrong)
        color_image_bgr = color_image_from_realsense
        draw_image = color_image_bgr.copy();
        overlay = draw_image.copy()

        # 6. Perform YOLO Inference
        results = model(color_image_bgr, verbose=False)

        # 7. Process Results & Extract KF Measurement
        measurement_3d = None;
        target_found_this_frame = False
        # Default measurements to NaN for plotting gaps when target isn't seen
        current_meas_x, current_meas_y = np.nan, np.nan

        if results and results[0].boxes is not None and results[0].masks is not None:
            # Move results to CPU for processing
            boxes = results[0].boxes.xyxy.cpu().numpy();
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy();
            masks_data = results[0].masks.data.cpu().numpy()
            current_img_shape_wh = (draw_image.shape[1], draw_image.shape[0])  # W, H

            # Resize masks only if necessary (e.g., if model predicts on different size)
            if masks_data.shape[1:] != (draw_image.shape[0], draw_image.shape[1]):
                masks_data = np.array(
                    [cv2.resize(m, current_img_shape_wh, interpolation=cv2.INTER_NEAREST) for m in masks_data])

            # Iterate through all detections
            for i in range(len(boxes)):
                conf = confs[i]
                if conf < CONFIDENCE_THRESHOLD: continue  # Skip low confidence detections

                x1, y1, x2, y2 = map(int, boxes[i]);
                cls_id = int(clss[i]);
                class_name = coco_names[cls_id]
                cx = (x1 + x2) // 2;
                cy = (y1 + y2) // 2  # BBox center
                X, Y, Z = 0.0, 0.0, 0.0;
                depth_value = 0.0  # Initialize 3D coords

                # Try to get depth and calculate 3D position
                try:
                    # Check if center pixel is within depth frame bounds
                    if 0 <= cx < CAM_WIDTH and 0 <= cy < CAM_HEIGHT:
                        depth_value = depth_frame.get_distance(cx, cy)
                        # Check if depth is within a reasonable range
                        if 0.1 < depth_value < 5.0:  # meters
                            X, Y, Z = convert_depth_to_phys_coord_using_realsense_intrinsics(cx, cy, depth_value,
                                                                                             color_intrinsics)
                        else:
                            Z = 0.0  # Mark as invalid Z if depth is unreasonable
                except Exception:
                    Z = 0.0  # Mark as invalid Z on error

                # Check if this is the target object and we haven't found it yet this frame
                if not target_found_this_frame and class_name == TARGET_CLASS_NAME and Z > 0:  # Check Z > 0 for valid 3D coord
                    measurement_3d = np.array([X, Y, Z])  # Use the first valid target's 3D coords
                    target_found_this_frame = True
                    current_meas_x, current_meas_y = X, Y  # Store non-NaN values for plot

                # Draw bounding box for the current object
                cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw segmentation mask for the current object
                try:
                    if i < len(masks_data): overlay[masks_data[i].astype(bool)] = mask_colors[cls_id].tolist()
                except Exception as e_mask:
                    print(f"Mask draw error: {e_mask}")
                # Draw label for the current object
                label = f"{class_name} {conf:.2f}";
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(draw_image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Generate grasp candidates for the target class
                if target_found_this_frame and class_name == TARGET_CLASS_NAME:
                    # Assuming 'masks_data' contains the binary mask for the target object
                    target_mask = masks_data[i]  # Mask for this specific object

                    # Generate grasp candidates
                    grasp_candidates = grasp_generator.generate_grasp_candidates(
                        target_mask,
                        depth_frame=depth_frame,
                        intrinsics=color_intrinsics
                    )

                    # Visualize top grasp candidates on the image
                    for (p1, p2, score) in grasp_candidates[:3]:  # Visualize top 3 candidates
                        try:
                            # Project 3D points back to 2D image coordinates
                            u1, v1 = map(int, rs.rs2_project_point_to_pixel(color_intrinsics, p1))
                            u2, v2 = map(int, rs.rs2_project_point_to_pixel(color_intrinsics, p2))

                            # Draw grasp candidate line
                            cv2.line(draw_image, (u1, v1), (u2, v2), (0, 255, 255), 2)

                            # Annotate score
                            cv2.putText(draw_image, f"G:{score:.2f}", (u1, v1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        except Exception as e:
                            print(f"Grasp visualization error: {e}")

        # 8. Update Kalman Filter & Get State
        predicted_states_apriori, _ = apkf.update(measurement_3d)  # Update with measurement (or None)
        current_posterior_state, current_posterior_cov = apkf.get_current_estimate()
        current_filt_x = current_posterior_state[0];
        current_filt_y = current_posterior_state[1]
        epsilon = 1e-9  # Small value to prevent sqrt of zero or negative
        current_std_x = np.sqrt(max(epsilon, current_posterior_cov[0, 0]))
        current_std_y = np.sqrt(max(epsilon, current_posterior_cov[1, 1]))

        # --- Occlusion Logic for Plotting ---
        if not target_found_this_frame:
            if not is_occluded: is_occluded = True; occlusion_start_time = current_plot_time; print(
                f"Target '{TARGET_CLASS_NAME}' lost at {occlusion_start_time:.2f}s")
        else:  # Target was found
            if is_occluded:
                print(f"Target '{TARGET_CLASS_NAME}' reacquired at {current_plot_time:.2f}s")
                if occlusion_start_time is not None and occlusion_start_time < current_plot_time:
                    occlusion_spans.append((occlusion_start_time, current_plot_time))
                    # Add shading for the occlusion period on the plot
                    ax_x.axvspan(occlusion_start_time, current_plot_time, color='grey', alpha=0.3, zorder=-1)
                    ax_y.axvspan(occlusion_start_time, current_plot_time, color='grey', alpha=0.3, zorder=-1)
                is_occluded = False;
                occlusion_start_time = None
        # ------------------------------------

        # --- Store Data for Plots ---
        plot_time.append(current_plot_time)
        plot_meas_x.append(current_meas_x)  # Appends NaN if target not found
        plot_meas_y.append(current_meas_y)  # Appends NaN if target not found
        plot_filt_x.append(current_filt_x)  # Filtered state is always available
        plot_filt_y.append(current_filt_y)  # Filtered state is always available
        plot_std_x.append(current_std_x)  # Standard deviation is always available
        plot_std_y.append(current_std_y)  # Standard deviation is always available

        # Buffer management: remove oldest data point if buffer is full
        if len(plot_time) > PLOT_BUFFER_SIZE:
            plot_time.pop(0);
            plot_meas_x.pop(0);
            plot_meas_y.pop(0)
            plot_filt_x.pop(0);
            plot_filt_y.pop(0);
            plot_std_x.pop(0);
            plot_std_y.pop(0)
        # -------------------------

        # --- Update Plots Periodically ---
        # Reduces load by not redrawing matplotlib on every single frame
        if absolute_time - last_plot_update > PLOT_UPDATE_INTERVAL:
            # Plot X vs Time
            line_meas_x.set_data(plot_time, plot_meas_x)  # Time on X, Measurement X on Y
            line_filt_x.set_data(plot_time, plot_filt_x)  # Time on X, Filtered X on Y
            fill_cov_x = update_fill_between(ax_x, fill_cov_x, plot_time,
                                             np.array(plot_filt_x) - np.array(plot_std_x),  # Lower bound Y
                                             np.array(plot_filt_x) + np.array(plot_std_x),  # Upper bound Y
                                             color='blue', alpha=PLOT_ALPHA)
            ax_x.relim();
            ax_x.autoscale_view()  # Rescale axes

            # Plot Y vs Time
            line_meas_y.set_data(plot_time, plot_meas_y)  # Time on X, Measurement Y on Y
            line_filt_y.set_data(plot_time, plot_filt_y)  # Time on X, Filtered Y on Y
            fill_cov_y = update_fill_between(ax_y, fill_cov_y, plot_time,
                                             np.array(plot_filt_y) - np.array(plot_std_y),  # Lower bound Y
                                             np.array(plot_filt_y) + np.array(plot_std_y),  # Upper bound Y
                                             color='cyan', alpha=PLOT_ALPHA)
            ax_y.relim();
            ax_y.autoscale_view()  # Rescale axes

            # Redraw the plot figure
            try:
                if plt.fignum_exists(fig.number):  # Check figure still exists
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()  # Process drawing events
            except Exception as e_plot:
                print(f"Plot drawing error: {e_plot}")
            last_plot_update = absolute_time  # Reset timer for next plot update
        # -----------------------------

        # 9. Visualize Kalman Predictions on CV Window
        max_pred_step = apkf.N - 1
        for step in range(apkf.N):
            # Extract predicted state [X,Y,Z, Vx,Vy,Vz] for this future step
            current_pred_state = predicted_states_apriori[step * apkf.state_dim:(step + 1) * apkf.state_dim]
            pred_pos_3d = current_pred_state[0:3]  # Get predicted [X, Y, Z]
            # Project prediction back to image pixels if it's in front of the camera
            if pred_pos_3d[2] > 0:
                try:
                    pred_u, pred_v = map(int, rs.rs2_project_point_to_pixel(color_intrinsics, pred_pos_3d))
                    # Draw if prediction is within image bounds
                    if 0 <= pred_u < CAM_WIDTH and 0 <= pred_v < CAM_HEIGHT:
                        # Color points from blue (near future) to red (far future)
                        ratio = step / max_pred_step if max_pred_step > 0 else 0
                        pred_color = (int(255 * (1 - ratio)), 0, int(255 * ratio))  # BGR color
                        cv2.circle(draw_image, (pred_u, pred_v), 4, pred_color, -1)  # Draw prediction point
                except Exception:
                    pass  # Ignore projection errors (e.g., point too far, etc.)

        # 10. Combine Overlay and Add FPS Text
        alpha = 0.4;
        cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)
        # --- Calculate and Display FPS using dedicated timer ---
        frame_count += 1
        current_time_fps = time.time()  # Get current time for FPS calc
        elapsed_time_fps = current_time_fps - fps_start_time  # <<< CORRECTED: Use fps_start_time
        # Calculate FPS only after at least 1 second has passed
        if elapsed_time_fps >= 1.0:
            fps = frame_count / elapsed_time_fps
            frame_count = 0  # Reset frame count for next interval
            fps_start_time = current_time_fps  # <<< CORRECTED: Reset the FPS timer start time
        # Display the most recently calculated FPS value
        cv2.putText(draw_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # ------------------------------------------------------

        # 11. Display OpenCV Image
        # Resize image for display window if necessary
        if draw_image.shape[0] != DISPLAY_HEIGHT:
            scale = DISPLAY_HEIGHT / draw_image.shape[0]
            img_display = cv2.resize(draw_image, (int(draw_image.shape[1] * scale), DISPLAY_HEIGHT),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            img_display = draw_image
        cv2.imshow("Live Segmentation + Kalman Tracking + Plot", img_display)

        # 12. Exit Condition
        key = cv2.waitKey(1) & 0xFF  # waitKey allows window to refresh and checks for keys
        if key == ord('q'): print("Exit key pressed."); break

finally:
    # 13. Cleanup
    print("Stopping RealSense pipeline.")
    if 'pipeline' in locals() and pipeline: pipeline.stop()
    cv2.destroyAllWindows()
    # Keep Matplotlib plot open after OpenCV windows close
    if 'fig' in locals() and plt.fignum_exists(fig.number):
        print("Final plot displayed. Close plot window to exit fully.")
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep plot window open until manually closed
    else:
        print("Plot window not found or already closed.")
    print("Script finished.")

# --- END OF FILE live_detect_segment_kalman_plot.py ---