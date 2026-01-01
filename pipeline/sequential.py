import os
import time

from utils.image_io import load_image, save_image
from filters.grayscale import apply_grayscale
from filters.gaussian_blur import apply_gaussian_blur
from filters.sobel import apply_sobel
from filters.sharpen import apply_sharpen
from filters.brightness import adjust_brightness


def process_images_sequential(image_paths, output_dir):
    """
    Sequential image processing pipeline.

    Applies all filters in order to each image.
    """

    start_time = time.time()

    for idx, image_path in enumerate(image_paths, start=1):
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

        if idx % 20 == 0:
            print(f"[Sequential] Processed {idx} images")

    total_time = time.time() - start_time
    return total_time
