import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.image_utils import read_image, save_image
from filters.grayscale import apply_grayscale
from filters.gaussian_blur import apply_gaussian_blur
from filters.edge_detection import apply_sobel_edge
from filters.sharpen import apply_sharpen
from filters.brightness import adjust_brightness


def process_single_image(image_path, output_dir):
    """
    Worker function for ProcessPoolExecutor.
    Processes one image from start to finish.
    """
    image = read_image(image_path)

    gray = apply_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    edges = apply_sobel_edge(blurred)
    sharpened = apply_sharpen(edges)
    final_image = adjust_brightness(sharpened, value=20)

    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, filename)
    save_image(save_path, final_image)

    return filename


def process_images_processpool(image_paths, output_dir, num_workers):
    """
    Parallel image processing using ProcessPoolExecutor.
    """
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # Submit all tasks to executor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_single_image, path, output_dir)
            for path in image_paths
        ]

        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()  # We don't need return value here

    elapsed = time.time() - start_time
    print(f"[RESULT] ProcessPoolExecutor time ({num_workers} workers): {elapsed:.2f} seconds")

    return elapsed
