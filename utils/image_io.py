import os
import numpy as np
from PIL import Image


def load_image(image_path):
    """
    Load an image from disk and convert it to a NumPy array.

    The image is converted to RGB to ensure consistent processing.
    """

    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def save_image(image_array, output_path):
    """
    Save a NumPy image array back to disk.

    Creates directories if they do not exist.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image = Image.fromarray(image_array.astype("uint8"))
    image.save(output_path)
