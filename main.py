import os

from utils.dataset_loader import load_image_paths
from pipeline.sequential import process_images_sequential
from pipeline.multiprocessing_impl import process_images_multiprocessing
from pipeline.futures_impl import process_images_futures


if __name__ == "__main__":
    dataset_dir = os.path.join(
        "data", "input", "edamame", "images"
    )

    image_paths = load_image_paths(dataset_dir, limit=200)

    # ================= Sequential =================
    seq_output = os.path.join("data", "output", "sequential")
    print("[INFO] Starting sequential processing...")
    seq_time = process_images_sequential(image_paths, seq_output)
    print(f"[RESULT] Sequential time: {seq_time:.2f} seconds\n")

    # ================= Multiprocessing =================
    mp_output = os.path.join("data", "output", "multiprocessing")
    workers = 4

    print(f"[INFO] Starting multiprocessing with {workers} workers...")
    mp_time = process_images_multiprocessing(
        image_paths, mp_output, workers
    )
    print(f"[RESULT] Multiprocessing time ({workers} workers): {mp_time:.2f} seconds\n")

    # ================= concurrent.futures =================
    futures_output = os.path.join("data", "output", "futures")

    print(f"[INFO] Starting concurrent.futures with {workers} workers...")
    futures_time = process_images_futures(
        image_paths, futures_output, workers
    )
    print(f"[RESULT] Futures time ({workers} workers): {futures_time:.2f} seconds")
