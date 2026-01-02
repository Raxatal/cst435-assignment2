import cv2
import numpy as np

def apply_sobel_edge(image: np.ndarray) -> np.ndarray:
    """
    Apply Sobel filter to detect edges.
    
    Args:
        image (np.ndarray): Input image (grayscale recommended).
    
    Returns:
        np.ndarray: Edge-detected image.
    """
    # If input is color, convert to grayscale first
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = np.uint8(np.clip(grad_mag, 0, 255))
    
    return grad_mag
