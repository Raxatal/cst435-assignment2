import cv2
import numpy as np


def apply_sobel(image):
    """
    Apply Sobel edge detection to highlight image edges.

    The Sobel operator computes gradients in the X and Y directions.
    """

    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude of gradient
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize and convert back to uint8
    magnitude = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    )

    # Convert back to 3-channel image
    edge_3ch = cv2.cvtColor(
        magnitude.astype(np.uint8), cv2.COLOR_GRAY2RGB
    )

    return edge_3ch
