import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import matplotlib.pyplot as plt
from utils.dataset_loader import load_image_paths
from pipeline.sequential import process_images_sequential
from pipeline.multiprocessing_impl import process_images_multiprocessing
from pipeline.futures_impl import process_images_futures
import multiprocessing  # Important for Windows

# ================= CONFIG =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "input", "edamame", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "benchmark")
WORKER_COUNTS = [1, 2, 4]
LIMIT_IMAGES = 200
CSV_FILE = os.path.join(PROJECT_ROOT, "benchmark_results.csv")

# ================= MAIN =================
if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows

    # Load dataset
    image_paths = load_image_paths(DATASET_DIR, limit=LIMIT_IMAGES)

    # Sequential
    seq_output = os.path.join(OUTPUT_DIR, "sequential")
    os.makedirs(seq_output, exist_ok=True)
    print("[INFO] Running sequential pipeline...")
    seq_time = process_images_sequential(image_paths, seq_output)
    print(f"[RESULT] Sequential time: {seq_time:.2f} seconds\n")

    results = []

    # Benchmark parallel methods
    for workers in WORKER_COUNTS:
        # Multiprocessing
        mp_output = os.path.join(OUTPUT_DIR, f"multiprocessing_{workers}")
        os.makedirs(mp_output, exist_ok=True)
        print(f"[INFO] Running multiprocessing with {workers} workers...")
        mp_time = process_images_multiprocessing(image_paths, mp_output, workers)
        print(f"[RESULT] Multiprocessing time: {mp_time:.2f} seconds\n")
        results.append({
            "Method": "Multiprocessing",
            "Workers": workers,
            "Time": mp_time,
            "Speedup": seq_time / mp_time,
            "Efficiency": (seq_time / mp_time) / workers
        })

        # concurrent.futures
        fut_output = os.path.join(OUTPUT_DIR, f"futures_{workers}")
        os.makedirs(fut_output, exist_ok=True)
        print(f"[INFO] Running concurrent.futures with {workers} workers...")
        fut_time = process_images_futures(image_paths, fut_output, workers)
        print(f"[RESULT] Futures time: {fut_time:.2f} seconds\n")
        results.append({
            "Method": "Futures",
            "Workers": workers,
            "Time": fut_time,
            "Speedup": seq_time / fut_time,
            "Efficiency": (seq_time / fut_time) / workers
        })

    # Save CSV
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Workers", "Time", "Speedup", "Efficiency"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[INFO] Benchmark results saved to {CSV_FILE}")

    # Plot and save graphs
    GRAPH_DIR = os.path.join(PROJECT_ROOT, "benchmark")
    os.makedirs(GRAPH_DIR, exist_ok=True)

    methods = ["Multiprocessing", "Futures"]

    # --- 1. Execution Time vs Workers ---
    plt.figure(figsize=(8, 6))
    for method in methods:
        times = [r["Time"] for r in results if r["Method"] == method]
        plt.bar([str(w) + f" ({method})" for w in WORKER_COUNTS], times, label=method)
    plt.title("Execution Time vs Number of Workers")
    plt.xlabel("Workers")
    plt.ylabel("Time (s)")
    plt.grid(True, axis="y")
    plt.xticks(rotation=30)
    plt.legend()
    time_graph_file = os.path.join(GRAPH_DIR, "execution_time_vs_workers.png")
    plt.savefig(time_graph_file, dpi=300)
    print(f"[INFO] Execution time graph saved to {time_graph_file}")
    plt.show(block=True)  # Interactive window, allows left/right arrows

    # --- 2. Speedup vs Workers ---
    plt.figure(figsize=(8, 6))
    for method in methods:
        speedups = [r["Speedup"] for r in results if r["Method"] == method]
        plt.plot(WORKER_COUNTS, speedups, marker="o", label=method)
    plt.title("Speedup vs Number of Workers")
    plt.xlabel("Workers")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.xticks(WORKER_COUNTS)
    plt.legend()
    speedup_graph_file = os.path.join(GRAPH_DIR, "speedup_vs_workers.png")
    plt.savefig(speedup_graph_file, dpi=300)
    print(f"[INFO] Speedup graph saved to {speedup_graph_file}")
    plt.show(block=True)

    # --- 3. Efficiency vs Workers ---
    plt.figure(figsize=(8, 6))
    for method in methods:
        efficiency = [r["Efficiency"] for r in results if r["Method"] == method]
        plt.plot(WORKER_COUNTS, efficiency, marker="o", label=method)
    plt.title("Efficiency vs Number of Workers")
    plt.xlabel("Workers")
    plt.ylabel("Efficiency")
    plt.grid(True)
    plt.xticks(WORKER_COUNTS)
    plt.legend()
    eff_graph_file = os.path.join(GRAPH_DIR, "efficiency_vs_workers.png")
    plt.savefig(eff_graph_file, dpi=300)
    print(f"[INFO] Efficiency graph saved to {eff_graph_file}")
    plt.show(block=True)
