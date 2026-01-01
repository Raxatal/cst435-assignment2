"""
run_benchmark.py

Benchmark the image processing pipeline using three approaches:
1. Sequential execution
2. Multiprocessing
3. concurrent.futures (ProcessPoolExecutor)

This script measures execution time, calculates speedup and efficiency, saves results
to a CSV file, and generates three graphs (execution time, speedup, efficiency) with
numeric labels for easy reference.
"""

import os
import sys
import csv
import multiprocessing

# Add project root to path so we can import modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import the image processing pipeline modules
from utils.dataset_loader import load_image_paths
from pipeline.sequential import process_images_sequential
from pipeline.multiprocessing_impl import process_images_multiprocessing
from pipeline.futures_impl import process_images_futures

# Matplotlib for plotting graphs
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "input", "edamame", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "benchmark")
WORKER_COUNTS = [1, 2, 4]  # Number of workers to test
LIMIT_IMAGES = 200         # Only use first 200 images for testing
CSV_FILE = os.path.join(PROJECT_ROOT, "benchmark_results.csv")

# ================= MAIN SCRIPT =================
if __name__ == "__main__":
    # Required for Windows when using multiprocessing
    multiprocessing.freeze_support()

    # Make sure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load image paths
    try:
        image_paths = load_image_paths(DATASET_DIR, limit=LIMIT_IMAGES)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ================= SEQUENTIAL EXECUTION =================
    seq_output = os.path.join(OUTPUT_DIR, "sequential")
    os.makedirs(seq_output, exist_ok=True)

    print("[INFO] Running sequential pipeline...")
    seq_time = process_images_sequential(image_paths, seq_output)
    print(f"[RESULT] Sequential time: {seq_time:.2f} seconds\n")

    # ================= PARALLEL BENCHMARK =================
    results = []
    methods = {
        "Multiprocessing": process_images_multiprocessing,
        "Futures": process_images_futures
    }

    # Test each parallel method with different worker counts
    for method_name, method_func in methods.items():
        for workers in WORKER_COUNTS:
            out_dir = os.path.join(OUTPUT_DIR, f"{method_name.lower()}_{workers}")
            os.makedirs(out_dir, exist_ok=True)

            print(f"[INFO] Running {method_name} with {workers} workers...")
            if method_name == "Multiprocessing":
                time_taken = method_func(image_paths, out_dir, workers)
            else:  # concurrent.futures
                time_taken = method_func(image_paths, out_dir, workers)

            print(f"[RESULT] {method_name} time ({workers} workers): {time_taken:.2f} seconds\n")

            # Compute speedup and efficiency
            speedup = seq_time / time_taken
            efficiency = speedup / workers

            results.append({
                "Method": method_name,
                "Workers": workers,
                "Time": time_taken,
                "Speedup": speedup,
                "Efficiency": efficiency
            })

    # ================= SAVE RESULTS TO CSV =================
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Workers", "Time", "Speedup", "Efficiency"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[INFO] Benchmark results saved to {CSV_FILE}\n")

    # ================= PLOTTING =================
    GRAPH_DIR = os.path.join(PROJECT_ROOT, "benchmark")
    os.makedirs(GRAPH_DIR, exist_ok=True)

    figures = []  # Keep references to figures so they don't get garbage collected

    # --- Execution Time Bar Chart ---
    fig1 = plt.figure(figsize=(8, 6))
    for method in methods:
        times = [r["Time"] for r in results if r["Method"] == method]
        bars = plt.bar([f"{w} ({method})" for w in WORKER_COUNTS], times, label=method)
        # Label each bar with its numeric value
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=9)
    plt.title("Execution Time vs Number of Workers")
    plt.xlabel("Workers")
    plt.ylabel("Time (s)")
    plt.grid(True, axis="y")
    plt.xticks(rotation=30)
    plt.legend()
    time_graph_file = os.path.join(GRAPH_DIR, "execution_time_vs_workers.png")
    plt.savefig(time_graph_file, dpi=300)
    print(f"[INFO] Execution time graph saved to {time_graph_file}")
    figures.append(fig1)

    # --- Speedup Line Chart ---
    fig2 = plt.figure(figsize=(8, 6))
    for method in methods:
        speedups = [r["Speedup"] for r in results if r["Method"] == method]
        plt.plot(WORKER_COUNTS, speedups, marker="o", label=method)
        # Annotate each point
        for x, y in zip(WORKER_COUNTS, speedups):
            plt.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=9)
    plt.title("Speedup vs Number of Workers")
    plt.xlabel("Workers")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.xticks(WORKER_COUNTS)
    plt.legend()
    speedup_graph_file = os.path.join(GRAPH_DIR, "speedup_vs_workers.png")
    plt.savefig(speedup_graph_file, dpi=300)
    print(f"[INFO] Speedup graph saved to {speedup_graph_file}")
    figures.append(fig2)

    # --- Efficiency Line Chart ---
    fig3 = plt.figure(figsize=(8, 6))
    for method in methods:
        efficiency = [r["Efficiency"] for r in results if r["Method"] == method]
        plt.plot(WORKER_COUNTS, efficiency, marker="o", label=method)
        # Annotate each point
        for x, y in zip(WORKER_COUNTS, efficiency):
            plt.text(x, y + 0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=9)
    plt.title("Efficiency vs Number of Workers")
    plt.xlabel("Workers")
    plt.ylabel("Efficiency")
    plt.grid(True)
    plt.xticks(WORKER_COUNTS)
    plt.legend()
    eff_graph_file = os.path.join(GRAPH_DIR, "efficiency_vs_workers.png")
    plt.savefig(eff_graph_file, dpi=300)
    print(f"[INFO] Efficiency graph saved to {eff_graph_file}")
    figures.append(fig3)

    # --- SHOW ALL GRAPHS AT ONCE ---
    plt.show()  # All figures open in one interactive window; you can switch between them
