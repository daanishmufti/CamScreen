import cv2
import numpy as np

def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    point_sums = pts.sum(axis=1)  # s = x + y
    point_differences = np.diff(pts, axis=1).ravel()  # d = y - x (ravel flattens the array)
    return np.array([
        pts[np.argmin(point_sums)],   # top-left
        pts[np.argmin(point_differences)],   # top-right
        pts[np.argmax(point_sums)],   # bottom-right
        pts[np.argmax(point_differences)],   # bottom-left
    ], dtype="float32")

def four_points_transform(image, pts):
    ordered_points = order_points(pts)
    (top_left, top_right, bottom_right, bottom_left) = ordered_points
    bottom_width = np.linalg.norm(bottom_right - bottom_left) # distance between bottom-right and bottom-left
    top_width = np.linalg.norm(top_right - top_left) # distance between top-right and top-left
    output_width = max(int(bottom_width), int(top_width))
    right_height = np.linalg.norm(top_right - bottom_right) # distance between top-right and bottom-right 
    left_height = np.linalg.norm(top_left - bottom_left) # distance between top-left and bottom-left
    output_height = max(int(right_height), int(left_height))
    destination_points = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32") # This defines where the screen corners should map to.

  # (0,0) -------- (W-1,0)
  # |                |
  # |                |
  # (0,H-1) ---- (W-1,H-1)
  # The above points define the corners of the output image, which will be a rectangle of size maxWidth x maxHeight.
  
    transform_matrix = cv2.getPerspectiveTransform(ordered_points, destination_points) # Compute the perspective transform matrix (map original to destination)  
    warped_image = cv2.warpPerspective(image, transform_matrix, (output_width, output_height)) # Apply the perspective transformation
    return warped_image
