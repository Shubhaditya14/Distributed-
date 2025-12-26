"""
Benchmark 3: Scalability Analysis

Show how system scales with worker count:
- Worker counts: 1, 2, 4, 8 workers
- Measures throughput, iteration time, parallel efficiency
- Calculates speedup compared to single worker
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (
    SRC_DIR, RESULTS_DIR,
    save_results, calculate_stats, get_system_info, clean_checkpoints
)


class ScalabilityBenchmark:
    """Benchmark for measuring scalability with different worker counts."""

    def __init__(self, num_iterations: int = 50, checkpoint_interval: int = 999):
        self.num_iterations = num_iterations
        self.checkpoint_interval = checkpoint_interval  # High value = no checkpointing
        self.master_process: Optional[subprocess.Popen] = None
        self.worker_processes: List[subprocess.Popen] = []

    def start_cluster(self, num_workers: int) -> bool:
        """Start master and workers."""
        env = os.environ.copy()
        env["EXPECTED_WORKERS"] = str(num_workers)
        env["CHECKPOINT_INTERVAL"] = str(self.checkpoint_interval)
        env["NUM_ITERATIONS"] = str(self.num_iterations)

        # Start master
        self.master_process = subprocess.Popen(
            [sys.executable, str(SRC_DIR / "master.py")],
            cwd=str(SRC_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        time.sleep(2)

        # Start workers
        self.worker_processes = []
        for _ in range(num_workers):
            proc = subprocess.Popen(
                [sys.executable, str(SRC_DIR / "worker.py")],
                cwd=str(SRC_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            self.worker_processes.append(proc)
            time.sleep(0.5)

        return True

    def wait_for_completion(self, timeout: float = 300) -> bool:
        """Wait for training to complete."""
        start = time.time()
        while time.time() - start < timeout:
            all_done = all(p.poll() is not None for p in self.worker_processes)
            if all_done:
                return True
            time.sleep(1)
        return False

    def stop_all(self):
        """Stop all processes."""
        for proc in self.worker_processes:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

        if self.master_process:
            try:
                self.master_process.kill()
                self.master_process.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

        self.worker_processes = []
        self.master_process = None

    def run_trial(self, num_workers: int) -> Dict:
        """Run a single scalability trial."""
        print(f"\n{'='*50}")
        print(f"Running trial: {num_workers} workers")
        print(f"{'='*50}")

        clean_checkpoints()

        start_time = time.time()
        self.start_cluster(num_workers)

        completed = self.wait_for_completion(timeout=180)
        end_time = time.time()

        total_time = end_time - start_time
        self.stop_all()

        # Calculate metrics
        batch_size = 32  # From worker.py
        samples_per_iteration = batch_size * num_workers
        total_samples = samples_per_iteration * self.num_iterations

        result = {
            "num_workers": num_workers,
            "completed": completed,
            "total_time": total_time,
            "num_iterations": self.num_iterations,
            "iterations_per_second": self.num_iterations / total_time if total_time > 0 else 0,
            "avg_iteration_time": total_time / self.num_iterations if self.num_iterations > 0 else 0,
            "samples_per_second": total_samples / total_time if total_time > 0 else 0,
            "total_samples": total_samples,
            "batch_size_per_worker": batch_size,
            "effective_batch_size": samples_per_iteration
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Iterations/sec: {result['iterations_per_second']:.2f}")
        print(f"  Samples/sec: {result['samples_per_second']:.2f}")

        return result

    def run_benchmark(self, worker_counts: List[int] = None, num_trials: int = 3) -> Dict:
        """Run full scalability benchmark."""
        if worker_counts is None:
            worker_counts = [1, 2, 4]

        results = {
            "system_info": get_system_info(),
            "config": {
                "num_iterations": self.num_iterations,
                "checkpoint_interval": self.checkpoint_interval,
                "worker_counts": worker_counts,
                "num_trials": num_trials
            },
            "trials_by_workers": {},
            "summary": {},
            "scalability_analysis": {}
        }

        for num_workers in worker_counts:
            worker_key = str(num_workers)
            results["trials_by_workers"][worker_key] = []

            for trial_num in range(num_trials):
                print(f"\n\n{'#'*60}")
                print(f"{num_workers} Workers, Trial {trial_num + 1}/{num_trials}")
                print(f"{'#'*60}")

                trial_result = self.run_trial(num_workers)
                trial_result["trial_number"] = trial_num + 1
                results["trials_by_workers"][worker_key].append(trial_result)

                time.sleep(2)

        # Calculate summary statistics
        for worker_key, trials in results["trials_by_workers"].items():
            valid_trials = [t for t in trials if t.get("completed", False)]
            if valid_trials:
                results["summary"][worker_key] = {
                    "total_time": calculate_stats([t["total_time"] for t in valid_trials]),
                    "iterations_per_second": calculate_stats([t["iterations_per_second"] for t in valid_trials]),
                    "samples_per_second": calculate_stats([t["samples_per_second"] for t in valid_trials]),
                    "avg_iteration_time": calculate_stats([t["avg_iteration_time"] for t in valid_trials])
                }

        # Calculate speedup and efficiency
        baseline_key = "1"
        if baseline_key in results["summary"]:
            baseline_time = results["summary"][baseline_key]["total_time"]["mean"]

            for worker_key, stats in results["summary"].items():
                num_workers = int(worker_key)
                current_time = stats["total_time"]["mean"]

                # Speedup = T(1) / T(N)
                speedup = baseline_time / current_time if current_time > 0 else 0
                # Efficiency = Speedup / N * 100
                efficiency = (speedup / num_workers) * 100 if num_workers > 0 else 0
                # Ideal speedup = N
                ideal_speedup = num_workers

                results["scalability_analysis"][worker_key] = {
                    "num_workers": num_workers,
                    "speedup": speedup,
                    "ideal_speedup": ideal_speedup,
                    "efficiency_percent": efficiency,
                    "overhead_factor": ideal_speedup / speedup if speedup > 0 else float('inf')
                }

        return results


def main():
    print("=" * 60)
    print("Scalability Benchmark")
    print("=" * 60)

    benchmark = ScalabilityBenchmark(num_iterations=30, checkpoint_interval=999)

    results = benchmark.run_benchmark(
        worker_counts=[1, 2, 4],
        num_trials=2
    )

    save_results("scalability_metrics.json", results)

    print("\n" + "=" * 60)
    print("Scalability Summary")
    print("=" * 60)

    print(f"\n{'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 50)

    for worker_key, analysis in results.get("scalability_analysis", {}).items():
        time_stats = results["summary"][worker_key]["total_time"]
        print(f"{analysis['num_workers']:<10} {time_stats['mean']:<12.2f} "
              f"{analysis['speedup']:<10.2f} {analysis['efficiency_percent']:<12.1f}%")

    return results


if __name__ == "__main__":
    main()
