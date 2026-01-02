import cv2
import numpy as np

def adjust_brightness(image: np.ndarray, value: int = 30) -> np.ndarray:
    """
    Adjust the brightness of an image.
    
    Args:
        image (np.ndarray): Input image.
        value (int): Positive to increase brightness, negative to decrease.
    
    Returns:
        np.ndarray: Brightness-adjusted image.
    """
    # Convert to float to avoid clipping during addition
    img_float = image.astype(np.float32)
    img_float += value
    img_float = np.clip(img_float, 0, 255)  # Ensure valid pixel range
    return img_float.astype(np.uint8)
