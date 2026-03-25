import numpy as np
import cv2
import settings

def filter_image(warped):
    if warped is None:
        return None
    if not settings.get("filter_enabled"):
        return warped  # filter disabled — pass the warped image through unchanged
    if len(warped.shape) == 3:
        gray_image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  # convert colour warped image to grayscale
    else:
        gray_image = warped  # already grayscale

    clip_limit = settings.get("clahe_clip")       # contrast clip limit for CLAHE
    tile_size = settings.get("clahe_tile")       # tile grid size (tile x tile blocks)
    sharpen_strength = settings.get("sharpen_strength")  # centre value of the sharpening kernel

    clahe_filter = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))  # adaptive histogram equaliser per tile
    enhanced_image = clahe_filter.apply(gray_image)   # equalise contrast in each tile independently

    center_value = max(1, sharpen_strength)  # centre must be at least 1 to avoid net blurring
    sharpen_kernel = np.array([[0, -1, 0],
                                  [-1, center_value, -1],
                                  [0, -1, 0]])  # Laplacian-based kernel; higher centre = stronger sharpening
    sharpened_image = cv2.filter2D(enhanced_image, -1, sharpen_kernel)  # apply sharpening via 2D convolution

    return sharpened_image