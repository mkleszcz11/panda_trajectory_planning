import typing as t
import numpy as np

#TODO ANDERS HERE
# Functions are just placeholders, modify however you see it.
# It would be nice though, if detect_objects_and_get_transformation
# will still be here.

# ANDERS HERE
# NOTE:
# This code (entire app) is not working out of the box due to the absolute paths.
# Serch entire project and replace every:
# /marcin/
# with:
# /student-nrt-lab/ (or similar, whatever it is on our computer)
# 
 
class ObjectDetector:
    def __init__(self):
        pass

    def detect_objects_and_get_transformation(self, image) -> t.Tuple[bool, t.List[np.ndarray]]:
        is_object_detected = False
        object_transformation_matrix = []
        # PLACEHOLDER FOR LOGIC
        return is_object_detected, object_transformation_matrix
