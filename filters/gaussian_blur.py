import cv2
import numpy as np

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a Gaussian blur to smooth the image.
    
    Args:
        image (np.ndarray): Input image.
        kernel_size (int): Size of the Gaussian kernel (must be odd).
    
    Returns:
        np.ndarray: Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Kernel must be odd
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred
