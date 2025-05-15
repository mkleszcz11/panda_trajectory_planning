# --- START OF FILE capture_realsense_frame_yolo.py ---

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime

def setup_realsense_pipeline(request_max_res=True):
    """Sets up and starts the RealSense pipeline, returning pipeline, config, and color profile."""
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    try:
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
    except RuntimeError as e:
         print(f"Error resolving pipeline: {e}. No device connected?")
         return None, None, None, None

    if not device.sensors:
        print("No RealSense sensors found.")
        return None, None, None, None

    color_sensor = device.first_color_sensor()
    if not color_sensor:
        print("No color sensor found.")
        return None, None, None, None

    # Find best color profile
    best_profile = None
    if request_max_res:
        color_video_profiles = [p for p in color_sensor.get_stream_profiles() if p.stream_type() == rs.stream.color and p.is_video_stream_profile()]
        if not color_video_profiles:
             print("No color video stream profiles found.")
             return None, None, None, None
        best_profile = max(color_video_profiles,
                           key=lambda p: p.as_video_stream_profile().width() * p.as_video_stream_profile().height())
        max_width = best_profile.as_video_stream_profile().width()
        max_height = best_profile.as_video_stream_profile().height()
        fps = best_profile.fps()
        fmt = best_profile.format()
        print(f"Requesting MAX Color Resolution: {max_width}x{max_height} @ {fps}fps, Format: {fmt}")
        config.enable_stream(rs.stream.color, max_width, max_height, fmt, fps)
    else:
        # Default reasonable resolution if max isn't requested or fails
        print("Requesting Default Color Resolution: 640x480 @ 30fps")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # We'll get the actual profile after starting

    # Enable depth stream - choose a resolution compatible with color if possible, common is 640x480
    depth_sensor = device.first_depth_sensor()
    if not depth_sensor:
        print("No depth sensor found.")
        # You might choose to continue without depth depending on requirements
        return None, None, None, None

    # Example: Enable depth stream at 640x480. Adjust if needed.
    try:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        print("Enabled Depth Stream: 640x480 @ 30fps")
    except RuntimeError as e:
        print(f"Could not enable requested depth stream: {e}. Trying default...")
        try:
            config.enable_stream(rs.stream.depth) # Let realsense choose a default depth stream
            print("Enabled default depth stream.")
        except RuntimeError as e2:
            print(f"Failed to enable any depth stream: {e2}")
            return None, None, None, None


    # Start streaming
    print("Starting RealSense pipeline...")
    try:
        profile = pipeline.start(config)
        time.sleep(2) # Allow auto-exposure to settle
        print("RealSense pipeline started.")

        # Get the actual color profile being used
        active_color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(f"Using Color Stream: {active_color_profile.width()}x{active_color_profile.height()} @ {active_color_profile.fps()}fps")


        return pipeline, profile, active_color_profile

    except RuntimeError as e:
        print(f"Failed to start pipeline: {e}")
        return None, None, None

def get_aligned_frames(pipeline, align_to=rs.stream.color):
    """Waits for and returns aligned color and depth frames."""
    align = rs.align(align_to)
    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames(timeout_ms=5000) # Increased timeout
        if not frames:
            print("Timed out waiting for frames.")
            return None, None

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            print("Failed to get aligned frames (depth or color missing).")
            return None, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, aligned_depth_frame # Return color image and depth *frame* (frame has metadata like intrinsics, get_distance)

    except RuntimeError as e:
        print(f"Error during frame capture/alignment: {e}")
        return None, None


def save_frame(image, output_folder="captured_frames", prefix="color"):
     """Saves a single numpy image array to the specified folder."""
     if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        # print(f"Created output directory: {output_folder}") # Less verbose

     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
     filename = os.path.join(output_folder, f"{prefix}_{timestamp}.png")
     try:
        cv2.imwrite(filename, image)
        print(f"Saved: {filename}")
        return filename
     except Exception as e:
        print(f"Error saving {filename}: {e}")
        return None

# --- END OF FILE capture_realsense_frame_yolo.py ---