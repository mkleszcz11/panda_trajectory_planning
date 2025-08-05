# antipodal_grasp_planner.py
"""
This module provides the AntipodalGraspPlanner class for detecting potential
2D grasp points on an object's binary mask using an antipodal heuristic.
"""

import cv2
import numpy as np
import math


class AntipodalGraspPlanner:
    def __init__(self, max_gripper_opening_px,
                 angle_tolerance_deg=10,
                 min_grasp_width_px=5,
                 contour_approx_epsilon_factor=0.005,
                 normal_neighborhood_k=1,
                 dist_penalty_weight=0.05,  # New: Penalty for distance from COG
                 width_favor_narrow_weight=0.01):  # New: Factor to favor narrower grasps
        """
        Initializes the grasp planner.

        Args:
            max_gripper_opening_px (float): Max opening width of gripper in pixels.
            angle_tolerance_deg (float): Allowable deviation from 180 deg for antipodal normals.
            min_grasp_width_px (float): Min practical grasp width in pixels.
            contour_approx_epsilon_factor (float): Factor for cv2.approxPolyDP.
            normal_neighborhood_k (int): Defines the neighborhood size for normal calculation.
            dist_penalty_weight (float): Weight for penalizing grasps far from object centroid.
                                         Higher values penalize distance more.
            width_favor_narrow_weight (float): Weight for favoring narrower grasps.
                                               Higher values favor narrower grasps more.
                                               Set to 0 to not consider width this way.
        """
        self.max_gripper_opening_px = max_gripper_opening_px
        self.min_grasp_width_px = min_grasp_width_px
        self.angle_tolerance_rad = math.radians(angle_tolerance_deg)
        self.antipodal_dot_product_threshold = -math.cos(self.angle_tolerance_rad)
        self.contour_approx_epsilon_factor = contour_approx_epsilon_factor
        self.normal_neighborhood_k = normal_neighborhood_k
        self.dist_penalty_weight = dist_penalty_weight
        self.width_favor_narrow_weight = width_favor_narrow_weight

    def _calculate_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            if len(contour) > 0:
                return np.mean(contour.reshape(-1, 2), axis=0).astype(int)
            return None
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        return np.array([centroid_x, centroid_y])

    def _calculate_normals(self, contour_points, original_dense_contour):
        normals = []
        num_points = len(contour_points)
        user_k = self.normal_neighborhood_k

        if num_points < 3:
            return [None] * num_points

        for i in range(num_points):
            p_curr = contour_points[i]
            effective_k = user_k
            if num_points < 2 * user_k + 1 and user_k > 0:
                effective_k = max(1, (num_points - 1) // 2)

            if num_points < 3:  # Ensure k=1 is possible
                effective_k = 0  # This case should be caught by num_points < 3 check above, but for safety
                # if k=0, it implies using p_curr and p_next which is not central difference
                # Better to stick to k>=1 which means at least 3 points are needed

            # With effective_k, we need at least 2*effective_k + 1 points to avoid wrap-around issues
            # if effective_k is large relative to num_points.
            # However, the modulo arithmetic handles wrap-around. The main concern is distinct points.
            # The current effective_k logic (max(1, (num_points-1)//2)) tries to make it sensible.
            # If num_points=3, effective_k=1. prev_idx = (i-1+3)%3, next_idx = (i+1)%3. Valid.
            # If num_points=4, effective_k=1.
            # If num_points=5, user_k=10 -> effective_k=max(1,(5-1)//2)=2.
            #   p_prev_idx = (i-2+5)%5, p_next_idx = (i+2)%5. Valid.

            p_prev_k_idx = (i - effective_k + num_points) % num_points
            p_next_k_idx = (i + effective_k) % num_points

            if p_prev_k_idx == p_next_k_idx:
                normals.append(None)
                continue

            p_prev_k = contour_points[p_prev_k_idx]
            p_next_k = contour_points[p_next_k_idx]

            tangent_vec = np.array(p_next_k, dtype=np.float32) - np.array(p_prev_k, dtype=np.float32)
            norm = np.array([-tangent_vec[1], tangent_vec[0]], dtype=np.float32)
            norm_mag = np.linalg.norm(norm)

            if norm_mag < 1e-6:
                if effective_k > 1 and num_points >= 3:  # Try fallback only if k was > 1
                    p_prev_fallback_idx = (i - 1 + num_points) % num_points
                    p_next_fallback_idx = (i + 1) % num_points
                    if p_prev_fallback_idx != p_next_fallback_idx:
                        p_prev_fallback = contour_points[p_prev_fallback_idx]
                        p_next_fallback = contour_points[p_next_fallback_idx]
                        tangent_vec = np.array(p_next_fallback, dtype=np.float32) - np.array(p_prev_fallback,
                                                                                             dtype=np.float32)
                        norm = np.array([-tangent_vec[1], tangent_vec[0]], dtype=np.float32)
                        norm_mag = np.linalg.norm(norm)
                if norm_mag < 1e-6:
                    normals.append(None)
                    continue

            unit_norm = norm / norm_mag
            offset_from_p_curr = unit_norm * 1.0
            test_point_np = np.array(p_curr, dtype=np.float32) + offset_from_p_curr
            test_point = (float(test_point_np[0]), float(test_point_np[1]))

            if cv2.pointPolygonTest(original_dense_contour, test_point, False) >= 0:
                unit_norm = -unit_norm
            normals.append(unit_norm)
        return normals

    def _is_grasp_line_collision_free(self, p1, p2, original_dense_contour, binary_mask_shape):
        h, w = binary_mask_shape
        mid_point_f = (np.array(p1, dtype=float) + np.array(p2, dtype=float)) / 2.0
        mid_point_i = tuple(mid_point_f.astype(int))

        if not (0 <= mid_point_i[1] < h and 0 <= mid_point_i[0] < w): return False
        test_mid_point = (float(mid_point_f[0]), float(mid_point_f[1]))
        if cv2.pointPolygonTest(original_dense_contour, test_mid_point, False) < 0: return False

        num_samples = 10
        for i in range(1, num_samples):
            t = i / float(num_samples)
            pt_f = np.array(p1, dtype=float) * (1 - t) + np.array(p2, dtype=float) * t
            pt_i = tuple(pt_f.astype(int))
            if not (0 <= pt_i[1] < h and 0 <= pt_i[0] < w): return False
            test_pt_intermediate = (float(pt_f[0]), float(pt_f[1]))
            if cv2.pointPolygonTest(original_dense_contour, test_pt_intermediate, False) < 0: return False
        return True

    def find_grasps(self, binary_mask):
        if binary_mask is None or binary_mask.ndim != 2 or binary_mask.dtype != np.uint8:
            raise ValueError("binary_mask must be a 2D uint8 NumPy array.")

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], None

        original_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(original_contour) < self.min_grasp_width_px * 2:
            return [], None

        object_centroid = self._calculate_centroid(original_contour)
        if object_centroid is None:
            return [], None

        epsilon = self.contour_approx_epsilon_factor * cv2.arcLength(original_contour, True)
        approx_contour_points = cv2.approxPolyDP(original_contour, epsilon, True)

        if len(approx_contour_points) < 3: return [], object_centroid

        contour_points = approx_contour_points.reshape(-1, 2)
        num_points = len(contour_points)
        if num_points < 2:
            return [], object_centroid

        normals = self._calculate_normals(contour_points, original_contour)
        candidate_local_grasps = []

        for i in range(num_points):
            p1_abs = contour_points[i]
            n1 = normals[i]
            if n1 is None:
                continue

            for j in range(i + 1, num_points):
                p2_abs = contour_points[j]
                n2 = normals[j]
                if n2 is None:
                    continue

                dot_product = np.dot(n1, n2)
                if dot_product < self.antipodal_dot_product_threshold:
                    width = np.linalg.norm(p1_abs - p2_abs)
                    if self.min_grasp_width_px <= width <= self.max_gripper_opening_px:
                        if self._is_grasp_line_collision_free(p1_abs, p2_abs, original_contour, binary_mask.shape):
                            grasp_center_abs = (np.array(p1_abs, dtype=float) + np.array(p2_abs, dtype=float)) / 2.0

                            # --- New Scoring Components ---
                            parallelism_term = -dot_product  # Range [~0, 1], higher is better

                            dist_to_centroid = np.linalg.norm(grasp_center_abs - object_centroid)
                            centroid_dist_term = 1.0 / (1.0 + self.dist_penalty_weight * dist_to_centroid)

                            width_term = 1.0  # Default if not favoring narrow
                            if self.width_favor_narrow_weight > 0:  # Only apply if weight is positive
                                width_term = 1.0 / (1.0 + self.width_favor_narrow_weight * width)

                            score = parallelism_term * centroid_dist_term * width_term
                            # --- End New Scoring Components ---

                            dx_abs = float(p2_abs[0]) - float(p1_abs[0])
                            dy_abs = float(p2_abs[1]) - float(p1_abs[1])
                            initial_grasp_angle_rad = math.atan2(dy_abs, dx_abs)

                            candidate_local_grasps.append({
                                "p1_local": tuple((p1_abs - object_centroid).astype(int)),
                                "p2_local": tuple((p2_abs - object_centroid).astype(int)),
                                "center_local": tuple((grasp_center_abs - object_centroid).astype(float)),
                                "grasp_center_abs": tuple(grasp_center_abs.astype(int)),
                                # Store absolute center for debug/info
                                "dist_to_obj_centroid": dist_to_centroid,  # Store for debug/info
                                "width_px": width,
                                "initial_angle_rad": initial_grasp_angle_rad,
                                "dot_product": dot_product,
                                "score": score
                            })

        candidate_local_grasps.sort(key=lambda g: g["score"], reverse=True)  # Higher score is better
        return candidate_local_grasps, object_centroid

    def transform_grasp_to_image_space(self, local_grasp_dict, new_object_centroid):
        if local_grasp_dict is None or new_object_centroid is None: return None
        p1_local = np.array(local_grasp_dict["p1_local"])
        p2_local = np.array(local_grasp_dict["p2_local"])
        center_local = np.array(local_grasp_dict["center_local"])  # This is grasp_center relative to old centroid

        new_centroid_arr = np.array(new_object_centroid, dtype=float)

        p1_abs_f = p1_local.astype(float) + new_centroid_arr
        p2_abs_f = p2_local.astype(float) + new_centroid_arr

        # Recalculate absolute grasp center based on new p1, p2 for consistency
        center_abs_f = (p1_abs_f + p2_abs_f) / 2.0
        # Or transform the stored relative center:
        # center_abs_f = center_local.astype(float) + new_centroid_arr

        p1_abs = tuple(p1_abs_f.astype(int))
        p2_abs = tuple(p2_abs_f.astype(int))
        center_abs = tuple(center_abs_f.astype(int))

        dx = p2_abs_f[0] - p1_abs_f[0]
        dy = p2_abs_f[1] - p1_abs_f[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            new_angle_rad = local_grasp_dict.get("initial_angle_rad", 0.0)
        else:
            new_angle_rad = math.atan2(dy, dx)
        return {
            "p1": p1_abs, "p2": p2_abs, "center_px": center_abs,
            "width_px": local_grasp_dict["width_px"],
            "angle_rad": new_angle_rad, "score": local_grasp_dict["score"]  # Score is intrinsic
        }

    def visualize_grasps(self, image_to_draw_on, absolute_grasps_list, num_top_grasps=5, line_color=(0, 255, 0),
                         point_color=(255, 0, 0), center_color=(0, 0, 255), finger_color=(255, 255, 0),
                         object_centroid_abs=None):  # Added object_centroid_abs for visualization
        vis_image = image_to_draw_on.copy()

        if len(vis_image.shape) == 2 or (len(vis_image.shape) == 3 and vis_image.shape[2] == 1):
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        if object_centroid_abs is not None:
            cv2.circle(vis_image, tuple(object_centroid_abs.astype(int)), 5, (255, 0, 255),
                       -1)  # Magenta for object CoG

        for i, grasp in enumerate(absolute_grasps_list):
            if i >= num_top_grasps: break
            p1, p2, center, width = grasp["p1"], grasp["p2"], grasp["center_px"], grasp["width_px"]
            cv2.line(vis_image, p1, p2, line_color, 2)
            cv2.circle(vis_image, p1, 5, point_color, -1)
            cv2.circle(vis_image, p2, 5, point_color, -1)
            cv2.circle(vis_image, center, 3, center_color, -1)  # Grasp center

            # Optionally draw a line from grasp center to object centroid
            if object_centroid_abs is not None:
                cv2.line(vis_image, center, tuple(object_centroid_abs.astype(int)), (200, 200, 200), 1)

            finger_length = max(10, int(width * 0.3))
            dx = float(p2[0] - p1[0]);
            dy = float(p2[1] - p1[1])
            perp_dx = -dy;
            perp_dy = dx
            mag_perp = math.sqrt(perp_dx ** 2 + perp_dy ** 2)
            if mag_perp > 1e-6:
                perp_dx_norm, perp_dy_norm = perp_dx / mag_perp, perp_dy / mag_perp
            else:
                perp_dx_norm, perp_dy_norm = 0, 1
            f1_s1 = (int(p1[0] - perp_dx_norm * finger_length / 2), int(p1[1] - perp_dy_norm * finger_length / 2))
            f1_s2 = (int(p1[0] + perp_dx_norm * finger_length / 2), int(p1[1] + perp_dy_norm * finger_length / 2))
            cv2.line(vis_image, f1_s1, f1_s2, finger_color, 2)
            f2_s1 = (int(p2[0] - perp_dx_norm * finger_length / 2), int(p2[1] - perp_dy_norm * finger_length / 2))
            f2_s2 = (int(p2[0] + perp_dx_norm * finger_length / 2), int(p2[1] + perp_dy_norm * finger_length / 2))
            cv2.line(vis_image, f2_s1, f2_s2, finger_color, 2)
        return vis_image


# # Example Usage:
# if __name__ == '__main__':
#     mask_height, mask_width = 300, 400
#     sample_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
#     ellipse_major_axis = mask_width // 3
#     ellipse_minor_axis = mask_height // 6  # For ellipse, we want to grasp across this
#     ellipse_angle = 30
#     initial_ellipse_center = (mask_width // 2, mask_height // 2)
#     cv2.ellipse(sample_mask, initial_ellipse_center, (ellipse_major_axis, ellipse_minor_axis), ellipse_angle, 0, 360,
#                 255, -1)

#     MAX_GRIPPER_OPENING_PIXELS = 180
#     MIN_GRIPPER_WIDTH_PIXELS = 20
#     # For ellipses, we want high parallelism and narrow width, and close to center
#     ANGLE_TOLERANCE_DEGREES = 15  # Stricter for better parallelism
#     CONTOUR_EPSILON_FACTOR = 0.003  # More points on the contour for finer detail
#     NORMAL_NEIGHBORHOOD_K = 2  # Moderate k for smoothing on the more detailed contour

#     # Scoring weights
#     DIST_PENALTY_WEIGHT = 0.05  # Increase to penalize distance from CoG more
#     WIDTH_FAVOR_NARROW_WEIGHT = 1.0  # Increase to favor narrower grasps more (good for ellipses)
#     # Set to 0 if you don't want to explicitly favor narrow

#     planner = AntipodalGraspPlanner(
#         max_gripper_opening_px=MAX_GRIPPER_OPENING_PIXELS,
#         min_grasp_width_px=MIN_GRIPPER_WIDTH_PIXELS,
#         angle_tolerance_deg=ANGLE_TOLERANCE_DEGREES,
#         contour_approx_epsilon_factor=CONTOUR_EPSILON_FACTOR,
#         normal_neighborhood_k=NORMAL_NEIGHBORHOOD_K,
#         dist_penalty_weight=DIST_PENALTY_WEIGHT,
#         width_favor_narrow_weight=WIDTH_FAVOR_NARROW_WEIGHT
#     )

#     print(f"--- Finding grasps on initial mask (k={NORMAL_NEIGHBORHOOD_K}, eps_factor={CONTOUR_EPSILON_FACTOR}, "
#           f"angle_tol={ANGLE_TOLERANCE_DEGREES}, dist_w={DIST_PENALTY_WEIGHT}, narrow_w={WIDTH_FAVOR_NARROW_WEIGHT}) ---")

#     local_grasps, object_centroid_initial = planner.find_grasps(sample_mask.copy())
#     best_local_grasp_for_tracking = None

#     if local_grasps and object_centroid_initial is not None:
#         print(
#             f"Found {len(local_grasps)} potential local grasps relative to initial centroid {object_centroid_initial}.")
#         best_local_grasp_for_tracking = local_grasps[0]
#         grasps_to_visualize_initial = []
#         for i, lg in enumerate(local_grasps[:min(10, len(local_grasps))]):  # Show more grasps if available
#             print(f"  Grasp {i + 1}: Score={lg['score']:.4f}, Width={lg['width_px']:.1f}px, "
#                   f"Dot={lg['dot_product']:.3f}, DistCOG={lg['dist_to_obj_centroid']:.1f}")
#             # print(f"    P1_loc={lg['p1_local']}, P2_loc={lg['p2_local']}, AbsGraspCenter={lg['grasp_center_abs']}")
#             abs_grasp = planner.transform_grasp_to_image_space(lg, object_centroid_initial)
#             if abs_grasp: grasps_to_visualize_initial.append(abs_grasp)

#         display_image_initial = cv2.cvtColor(sample_mask, cv2.COLOR_GRAY2BGR)
#         display_image_with_grasps_initial = planner.visualize_grasps(
#             display_image_initial,
#             grasps_to_visualize_initial,
#             num_top_grasps=5,  # Visualize top 5
#             object_centroid_abs=object_centroid_initial  # Pass centroid for visualization
#         )
#         # cv2.imshow("Initial Mask with Grasps", display_image_with_grasps_initial)
#     else:
#         print("No suitable grasps found on the initial mask or centroid not found.")

#     # --- Simulation of object moving ---
#     if best_local_grasp_for_tracking and object_centroid_initial is not None:
#         print("\n--- Simulating object movement and tracking the best grasp ---")
#         new_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
#         delta_x, delta_y = 50, 30
#         new_ellipse_center_tracked = (initial_ellipse_center[0] + delta_x, initial_ellipse_center[1] + delta_y)
#         cv2.ellipse(new_mask, new_ellipse_center_tracked, (ellipse_major_axis, ellipse_minor_axis), ellipse_angle, 0,
#                     360, 255, -1)

#         new_contours_tracked, _ = cv2.findContours(new_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         object_centroid_tracked = None
#         if new_contours_tracked:
#             new_main_contour_tracked = max(new_contours_tracked, key=cv2.contourArea)
#             object_centroid_tracked = planner._calculate_centroid(new_main_contour_tracked)

#         if object_centroid_tracked is not None:
#             print(f"Object presumed moved. New centroid: {object_centroid_tracked}")
#             tracked_grasp_absolute = planner.transform_grasp_to_image_space(best_local_grasp_for_tracking,
#                                                                             object_centroid_tracked)
#             if tracked_grasp_absolute:
#                 print(
#                     f"  Tracked Grasp (absolute): P1={tracked_grasp_absolute['p1']}, P2={tracked_grasp_absolute['p2']}, "
#                     f"Width={tracked_grasp_absolute['width_px']:.2f}px, Angle={math.degrees(tracked_grasp_absolute['angle_rad']):.2f}deg, "
#                     f"Original Score={tracked_grasp_absolute['score']:.4f}")
#                 display_image_tracked = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR)
#                 display_image_with_tracked_grasp = planner.visualize_grasps(
#                     display_image_tracked, [tracked_grasp_absolute], num_top_grasps=1,
#                     object_centroid_abs=object_centroid_tracked
#                 )
#                 # cv2.imshow("Moved Object with Tracked Grasp", display_image_with_tracked_grasp)
#             else:
#                 print("Could not transform the tracked grasp.")
#         else:
#             print("Could not find centroid for the moved object in the new mask.")

#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     elif not best_local_grasp_for_tracking:
#         print("No initial best grasp was selected to track.")
#     elif object_centroid_initial is None:
#         print("Initial object centroid was not found.")