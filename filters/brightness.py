import numpy as np


def adjust_brightness(image, delta=30):
    """
    Adjust image brightness.

    Positive delta increases brightness.
    Negative delta decreases brightness.
    """

    if image is None:
        raise ValueError("Input image is None")

    # Convert to int to prevent overflow
    temp = image.astype(np.int16) + delta

    # Clip values to valid range
    temp = np.clip(temp, 0, 255)

    return temp.astype(np.uint8)
