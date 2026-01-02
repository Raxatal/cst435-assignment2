import sys
import os

# Add project root to path so we can import modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import csv
import matplotlib.pyplot as plt

from utils.dataset_loader import load_image_paths
from pipeline.sequential import process_images_sequential
from pipeline.multiprocessing_impl import process_images_multiprocessing
from pipeline.processpool_impl import process_images_processpool
from pipeline.threadpool_impl import process_images_threadpool

# ---------------- Configuration ----------------
DATASET_DIR = "data/input/images"       # Total 3000 images from 3 food classes
OUTPUT_BASE = "data/output"             # Output for processed images
LIMIT_IMAGES = 3000                     # Change this to switch between first 1000, 2000, 3000 images, etc
WORKER_COUNTS = [1, 2, 4]

RESULT_CSV = "benchmark/plots/benchmark_results.csv"
PLOT_DIR = "benchmark/plots"

os.makedirs(PLOT_DIR, exist_ok=True)


def compute_speedup(seq_time, parallel_time):
    return seq_time / parallel_time

def compute_efficiency(speedup, workers):
    return speedup / workers

def main():
    # Load image paths from dataset folder
    image_paths = load_image_paths(DATASET_DIR, limit=LIMIT_IMAGES)
    print(f"[INFO] Loaded {len(image_paths)} images")

    results = []

    # -------- Sequential baseline --------
    print("\n[INFO] Running sequential pipeline...")
    seq_output = os.path.join(OUTPUT_BASE, "sequential")
    seq_time = process_images_sequential(image_paths, seq_output)

    results.append({
        "method": "Sequential",
        "workers": 1,
        "time": seq_time,
        "speedup": 1.0,
        "efficiency": 1.0
    })

    # -------- Multiprocessing --------
    for workers in WORKER_COUNTS:
        print(f"\n[INFO] Running multiprocessing ({workers} workers)...")
        mp_output = os.path.join(OUTPUT_BASE, f"mp_{workers}")
        mp_time = process_images_multiprocessing(image_paths, mp_output, workers)

        results.append({
            "method": "Multiprocessing",
            "workers": workers,
            "time": mp_time,
            "speedup": compute_speedup(seq_time, mp_time),
            "efficiency": compute_efficiency(compute_speedup(seq_time, mp_time), workers)
        })

    # -------- ProcessPoolExecutor --------
    for workers in WORKER_COUNTS:
        print(f"\n[INFO] Running ProcessPoolExecutor ({workers} workers)...")
        pp_output = os.path.join(OUTPUT_BASE, f"process_pool_{workers}")
        pp_time = process_images_processpool(image_paths, pp_output, workers)

        results.append({
            "method": "ProcessPoolExecutor",
            "workers": workers,
            "time": pp_time,
            "speedup": compute_speedup(seq_time, pp_time),
            "efficiency": compute_efficiency(compute_speedup(seq_time, pp_time), workers)
        })

    # -------- ThreadPoolExecutor --------
    for workers in WORKER_COUNTS:
        print(f"\n[INFO] Running ThreadPoolExecutor ({workers} workers)...")
        th_output = os.path.join(OUTPUT_BASE, f"thread_pool_{workers}")
        th_time = process_images_threadpool(image_paths, th_output, workers)

        results.append({
            "method": "ThreadPoolExecutor",
            "workers": workers,
            "time": th_time,
            "speedup": compute_speedup(seq_time, th_time),
            "efficiency": compute_efficiency(compute_speedup(seq_time, th_time), workers)
        })

    # -------- Save CSV --------
    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "workers", "time", "speedup", "efficiency"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[INFO] Results saved to {RESULT_CSV}")

    generate_plots(results)


def generate_plots(results):
    methods = ["Multiprocessing", "ProcessPoolExecutor", "ThreadPoolExecutor"]

    # -------- Execution Time --------
    plt.figure()
    for method in methods:
        xs = [r["workers"] for r in results if r["method"] == method]
        ys = [r["time"] for r in results if r["method"] == method]
        plt.plot(xs, ys, marker="o", label=method)
        for x, y in zip(xs, ys):
            plt.text(x, y, f"{y:.2f}s", ha="center", va="bottom")

    plt.xlabel("Number of Workers")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time vs Workers")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/execution_time.png")

    # -------- Speedup --------
    plt.figure()
    for method in methods:
        xs = [r["workers"] for r in results if r["method"] == method]
        ys = [r["speedup"] for r in results if r["method"] == method]
        plt.plot(xs, ys, marker="o", label=method)
        for x, y in zip(xs, ys):
            plt.text(x, y, f"{y:.2f}", ha="center", va="bottom")

    plt.xlabel("Number of Workers")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Workers")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/speedup.png")

    # -------- Efficiency --------
    plt.figure()
    for method in methods:
        xs = [r["workers"] for r in results if r["method"] == method]
        ys = [r["efficiency"] for r in results if r["method"] == method]
        plt.plot(xs, ys, marker="o", label=method)
        for x, y in zip(xs, ys):
            plt.text(x, y, f"{y:.2f}", ha="center", va="bottom")

    plt.xlabel("Number of Workers")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Workers")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/efficiency.png")

    plt.show()


if __name__ == "__main__":
    main()
