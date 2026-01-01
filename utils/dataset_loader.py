import os


def load_image_paths(
    base_dir,
    limit=200,
    extensions=(".jpg", ".jpeg", ".png")
):
    """
    Load image file paths from a given directory.

    Parameters:
    - base_dir: directory containing image files
    - limit: maximum number of images to load
    - extensions: valid image extensions

    Returns:
    - List of absolute image file paths
    """

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    image_files = []

    # Sort files to ensure consistent ordering across runs
    for filename in sorted(os.listdir(base_dir)):
        if filename.lower().endswith(extensions):
            image_files.append(os.path.join(base_dir, filename))

        if len(image_files) >= limit:
            break

    if len(image_files) == 0:
        raise RuntimeError("No images found in dataset directory.")

    print(f"[INFO] Loaded {len(image_files)} images from {base_dir}")
    return image_files
