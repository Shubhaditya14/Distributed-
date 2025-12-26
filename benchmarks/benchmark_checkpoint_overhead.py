"""
Benchmark 2: Checkpoint Overhead

Quantify performance impact of checkpointing at different frequencies:
- No checkpointing (baseline)
- Every 50 iterations
- Every 20 iterations
- Every 10 iterations (current default)
- Every 5 iterations

Measures:
- Total training time
- Average iteration time
- Checkpoint save time
- Memory usage during checkpoint
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Optional
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (
    SRC_DIR, RESULTS_DIR,
    save_results, calculate_stats, get_system_info, clean_checkpoints
)


class CheckpointOverheadBenchmark:
    """Benchmark checkpoint overhead at different frequencies."""

    def __init__(self, num_workers: int = 4, num_iterations: int = 100):
        self.num_workers = num_workers
        self.num_iterations = num_iterations
        self.master_process: Optional[subprocess.Popen] = None
        self.worker_processes: List[subprocess.Popen] = []

    def start_cluster(self, checkpoint_interval: int) -> bool:
        """Start master and workers with specified checkpoint interval."""
        env = os.environ.copy()
        env["EXPECTED_WORKERS"] = str(self.num_workers)
        env["CHECKPOINT_INTERVAL"] = str(checkpoint_interval)
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
        for _ in range(self.num_workers):
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
            # Check if all workers have finished
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

    def count_checkpoints(self) -> int:
        """Count number of checkpoint files created."""
        checkpoint_dir = Path("/tmp/checkpoints")
        if not checkpoint_dir.exists():
            return 0

        count = 0
        for rank_dir in checkpoint_dir.iterdir():
            if rank_dir.is_dir():
                count += len(list(rank_dir.glob("*.pt")))
        return count

    def get_checkpoint_size(self) -> float:
        """Get total size of checkpoints in MB."""
        checkpoint_dir = Path("/tmp/checkpoints")
        if not checkpoint_dir.exists():
            return 0.0

        total_size = 0
        for rank_dir in checkpoint_dir.iterdir():
            if rank_dir.is_dir():
                for f in rank_dir.glob("*.pt"):
                    total_size += f.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB

    def run_trial(self, checkpoint_interval: int) -> Dict:
        """Run a single trial with specified checkpoint interval."""
        print(f"\n{'='*50}")
        print(f"Running trial: checkpoint_interval={checkpoint_interval}")
        print(f"{'='*50}")

        clean_checkpoints()

        start_time = time.time()
        self.start_cluster(checkpoint_interval)

        completed = self.wait_for_completion(timeout=180)
        end_time = time.time()

        total_time = end_time - start_time
        num_checkpoints = self.count_checkpoints()
        checkpoint_size = self.get_checkpoint_size()

        self.stop_all()

        # Calculate metrics
        avg_iteration_time = total_time / self.num_iterations if self.num_iterations > 0 else 0
        expected_checkpoints = (self.num_iterations // checkpoint_interval) * self.num_workers if checkpoint_interval > 0 else 0

        result = {
            "checkpoint_interval": checkpoint_interval,
            "completed": completed,
            "total_time": total_time,
            "num_iterations": self.num_iterations,
            "avg_iteration_time": avg_iteration_time,
            "num_checkpoints": num_checkpoints,
            "expected_checkpoints": expected_checkpoints,
            "total_checkpoint_size_mb": checkpoint_size,
            "avg_checkpoint_size_mb": checkpoint_size / max(num_checkpoints, 1)
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg iteration time: {avg_iteration_time:.4f}s")
        print(f"  Checkpoints created: {num_checkpoints}")
        print(f"  Total checkpoint size: {checkpoint_size:.2f} MB")

        return result

    def run_benchmark(self, checkpoint_intervals: List[int] = None, num_trials: int = 3) -> Dict:
        """Run full benchmark across all checkpoint intervals."""
        if checkpoint_intervals is None:
            checkpoint_intervals = [0, 50, 20, 10, 5]  # 0 = no checkpointing

        results = {
            "system_info": get_system_info(),
            "config": {
                "num_workers": self.num_workers,
                "num_iterations": self.num_iterations,
                "checkpoint_intervals": checkpoint_intervals,
                "num_trials": num_trials
            },
            "trials_by_interval": {},
            "summary": {}
        }

        for interval in checkpoint_intervals:
            interval_key = str(interval) if interval > 0 else "none"
            results["trials_by_interval"][interval_key] = []

            for trial_num in range(num_trials):
                print(f"\n\n{'#'*60}")
                print(f"Interval {interval_key}, Trial {trial_num + 1}/{num_trials}")
                print(f"{'#'*60}")

                trial_result = self.run_trial(interval if interval > 0 else 999999)
                trial_result["trial_number"] = trial_num + 1
                results["trials_by_interval"][interval_key].append(trial_result)

                time.sleep(2)

        # Calculate summary statistics
        for interval_key, trials in results["trials_by_interval"].items():
            valid_trials = [t for t in trials if t.get("completed", False)]
            if valid_trials:
                total_times = [t["total_time"] for t in valid_trials]
                results["summary"][interval_key] = {
                    "total_time": calculate_stats(total_times),
                    "avg_iteration_time": calculate_stats([t["avg_iteration_time"] for t in valid_trials]),
                    "num_checkpoints": valid_trials[0]["num_checkpoints"],
                    "checkpoint_size_mb": valid_trials[0]["total_checkpoint_size_mb"]
                }

        # Calculate overhead compared to baseline
        baseline_key = "none"
        if baseline_key in results["summary"]:
            baseline_time = results["summary"][baseline_key]["total_time"]["mean"]
            for interval_key, stats in results["summary"].items():
                if interval_key != baseline_key:
                    overhead_time = stats["total_time"]["mean"] - baseline_time
                    overhead_pct = (overhead_time / baseline_time) * 100 if baseline_time > 0 else 0
                    stats["overhead_seconds"] = overhead_time
                    stats["overhead_percent"] = overhead_pct

        return results


def main():
    print("=" * 60)
    print("Checkpoint Overhead Benchmark")
    print("=" * 60)

    benchmark = CheckpointOverheadBenchmark(num_workers=4, num_iterations=50)

    # Run with fewer trials for speed
    results = benchmark.run_benchmark(
        checkpoint_intervals=[0, 25, 10, 5],
        num_trials=2
    )

    save_results("checkpoint_overhead.json", results)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for interval, stats in results.get("summary", {}).items():
        print(f"\nCheckpoint interval: {interval}")
        print(f"  Total time: {stats['total_time']['mean']:.2f}s (+/- {stats['total_time']['std']:.2f})")
        if "overhead_percent" in stats:
            print(f"  Overhead: {stats['overhead_percent']:.1f}%")

    return results


if __name__ == "__main__":
    main()
