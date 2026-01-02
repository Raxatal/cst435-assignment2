import os
from typing import List

def load_image_paths(base_dir: str, limit: int = None) -> List[str]:
    """
    Load full paths to all images in the specified dataset folder.
    
    Args:
        base_dir (str): Path to the folder containing images
        limit (int, optional): Max number of images to load. Loads all if None

    Returns:
        List[str]: List of full image paths
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    # Collect all files that are likely images
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_paths = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.lower().endswith(valid_extensions)
    ]

    # Sort so results are deterministic
    image_paths.sort()

    # Apply limit if got
    if limit:
        image_paths = image_paths[:limit]

    print(f"[INFO] Loaded {len(image_paths)} images from {base_dir}")
    return image_paths
