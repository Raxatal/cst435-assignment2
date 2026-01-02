import cv2
import numpy as np

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB/BGR image to grayscale using standard luminance formula.
    
    Args:
        image (np.ndarray): Input BGR image.
    
    Returns:
        np.ndarray: Grayscale image.
    """
    # OpenCV handles the conversion efficiently
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray
