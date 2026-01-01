import os
import time
from concurrent.futures import ProcessPoolExecutor

from utils.image_io import load_image, save_image
from filters.grayscale import apply_grayscale
from filters.gaussian_blur import apply_gaussian_blur
from filters.sobel import apply_sobel
from filters.sharpen import apply_sharpen
from filters.brightness import adjust_brightness


def process_single_image(image_path, output_dir):
    """
    Worker function for ProcessPoolExecutor.
    """
    image = load_image(image_path)

    image = apply_grayscale(image)
    image = apply_gaussian_blur(image)
    image = apply_sobel(image)
    image = apply_sharpen(image)
    image = adjust_brightness(image, delta=30)

    output_path = os.path.join(
        output_dir,
        os.path.basename(image_path)
    )
    save_image(image, output_path)


def process_images_futures(image_paths, output_dir, num_workers):
    """
    Run image processing using concurrent.futures ProcessPoolExecutor.
    """

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_single_image, img_path, output_dir)
            for img_path in image_paths
        ]

        # Ensure all tasks complete
        for future in futures:
            future.result()

    total_time = time.time() - start_time
    return total_time
