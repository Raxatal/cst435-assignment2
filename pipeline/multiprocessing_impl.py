import os
import time

from utils.image_io import load_image, save_image
from filters.grayscale import apply_grayscale
from filters.gaussian_blur import apply_gaussian_blur
from filters.sobel import apply_sobel
from filters.sharpen import apply_sharpen
from filters.brightness import adjust_brightness


def process_single_image(args):
    """
    Worker function for multiprocessing.

    Each process handles one image independently.
    """
    image_path, output_dir = args

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


def process_images_multiprocessing(image_paths, output_dir, num_workers):
    """
    Run the image processing pipeline using multiprocessing.
    """

    from multiprocessing import Pool

    start_time = time.time()

    # Prepare arguments for each worker
    tasks = [
        (image_path, output_dir)
        for image_path in image_paths
    ]

    with Pool(processes=num_workers) as pool:
        pool.map(process_single_image, tasks)

    total_time = time.time() - start_time
    return total_time
