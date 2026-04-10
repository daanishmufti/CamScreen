import cv2
import numpy as np


def order_points(pts): # takes 4 points and orders them in consistent way: top-left, top-right, bottom-right, bottom-left
    pts = pts.reshape(4, 2).astype("float32")
    point_sums = pts.sum(axis=1)
    point_differences = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(point_sums)],
        pts[np.argmin(point_differences)],
        pts[np.argmax(point_sums)],
        pts[np.argmax(point_differences)],
    ], dtype="float32")


def four_points_transform(image, pts): # applies a perspective transform to the image using the given 4 points
    ordered_points = order_points(pts)
    (top_left, top_right, bottom_right, bottom_left) = ordered_points
    output_width = max(int(np.linalg.norm(bottom_right - bottom_left)), int(np.linalg.norm(top_right - top_left)))
    output_height = max(int(np.linalg.norm(top_right - bottom_right)), int(np.linalg.norm(top_left - bottom_left)))
    destination_points = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")
    transform_matrix = cv2.getPerspectiveTransform(ordered_points, destination_points)
    return cv2.warpPerspective(image, transform_matrix, (output_width, output_height))
