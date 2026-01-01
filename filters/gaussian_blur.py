import cv2
import numpy as np


def apply_gaussian_blur(image):
    """
    Apply a 3x3 Gaussian blur to smooth the image.

    This helps reduce noise before edge detection.
    OpenCV is used here for efficiency and clarity.
    """

    if image is None:
        raise ValueError("Input image is None")

    # Kernel size (3x3) with standard deviation automatically computed
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    return blurred.astype(np.uint8)
