import numpy as np


def apply_grayscale(image):
    """
    Convert an RGB image to grayscale using the luminance formula.

    Formula:
    Y = 0.299R + 0.587G + 0.114B

    This reduces the image to a single intensity channel while
    preserving perceived brightness.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image")

    # Split channels
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # Luminance calculation
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Convert back to 3-channel grayscale for consistency
    gray_3ch = np.stack((gray, gray, gray), axis=2)

    return gray_3ch.astype(np.uint8)
