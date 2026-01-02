import os
import time

from utils.image_utils import read_image, save_image
from filters.grayscale import apply_grayscale
from filters.gaussian_blur import apply_gaussian_blur
from filters.edge_detection import apply_sobel_edge
from filters.sharpen import apply_sharpen
from filters.brightness import adjust_brightness


def process_images_sequential(image_paths, output_dir):
    """
    Process images one by one using a sequential pipeline.
    This serves as the baseline for all parallel comparisons.
    """
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    for idx, image_path in enumerate(image_paths, start=1):
        # Read image from disk
        image = read_image(image_path)

        # Apply image processing pipeline
        gray = apply_grayscale(image)
        blurred = apply_gaussian_blur(gray)
        edges = apply_sobel_edge(blurred)
        sharpened = apply_sharpen(edges)
        final_image = adjust_brightness(sharpened, value=20)

        # Save output
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, filename)
        save_image(save_path, final_image)

        # Progress update every 50 images
        if idx % 50 == 0:
            print(f"[Sequential] Processed {idx} images")

    elapsed = time.time() - start_time
    print(f"[RESULT] Sequential time: {elapsed:.2f} seconds")

    return elapsed
