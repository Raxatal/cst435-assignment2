import cv2
import numpy as np


def apply_sharpen(image):
    """
    Sharpen the image using a convolution kernel.

    This enhances edges and fine details after smoothing.
    """

    if image is None:
        raise ValueError("Input image is None")

    # Sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened.astype(np.uint8)
