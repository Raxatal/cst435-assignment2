import cv2
import numpy as np

def read_image(path: str):
    """Read an image from disk as a BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img

def save_image(path: str, image: np.ndarray):
    """Save a numpy array as an image to disk."""
    cv2.imwrite(path, image)
