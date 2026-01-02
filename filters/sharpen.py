import cv2
import numpy as np

def apply_sharpen(image: np.ndarray) -> np.ndarray:
    """
    Enhance edges to sharpen the image.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Sharpened image.
    """
    # Simple sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened
