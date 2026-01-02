import os
import time
from multiprocessing import Pool

from utils.image_utils import read_image, save_image
from filters.grayscale import apply_grayscale
from filters.gaussian_blur import apply_gaussian_blur
from filters.edge_detection import apply_sobel_edge
from filters.sharpen import apply_sharpen
from filters.brightness import adjust_brightness


def process_single_image(args):
    """
    Worker function.
    Each process handles exactly one image.
    """
    image_path, output_dir = args

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


def process_images_multiprocessing(image_paths, output_dir, num_workers):
    """
    Parallel image processing using multiprocessing.Pool.
    """
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # Prepare arguments for workers
    worker_args = [(path, output_dir) for path in image_paths]

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image, worker_args)

    elapsed = time.time() - start_time
    print(f"[RESULT] Multiprocessing time ({num_workers} workers): {elapsed:.2f} seconds")

    return elapsed
