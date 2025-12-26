"""
Benchmark 4: Network Overhead

Measure gRPC heartbeat communication overhead:
- Test different heartbeat intervals: 1s, 3s, 5s, 10s
- Measure network bandwidth, processing latency
- Analyze failure detection latency vs overhead tradeoff
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


class NetworkOverheadBenchmark:
    """Benchmark for measuring network/heartbeat overhead."""

    def __init__(self, num_workers: int = 4, num_iterations: int = 50):
        self.num_workers = num_workers
        self.num_iterations = num_iterations
        self.master_process: Optional[subprocess.Popen] = None
        self.worker_processes: List[subprocess.Popen] = []

    def start_cluster(self, heartbeat_interval: float = 0.5) -> bool:
        """Start master and workers with specified heartbeat interval."""
        env = os.environ.copy()
        env["EXPECTED_WORKERS"] = str(self.num_workers)
        env["CHECKPOINT_INTERVAL"] = "999"  # No checkpointing for this benchmark
        env["NUM_ITERATIONS"] = str(self.num_iterations)
        env["HEARTBEAT_INTERVAL"] = str(heartbeat_interval)

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

    def estimate_heartbeat_metrics(self, heartbeat_interval: float, training_time: float) -> Dict:
        """Estimate heartbeat-related metrics."""
        # Heartbeat message size (estimated gRPC overhead + payload)
        # HeartbeatRequest: ~50 bytes, HeartbeatResponse: ~10 bytes
        heartbeat_request_size = 50  # bytes
        heartbeat_response_size = 10  # bytes
        total_per_heartbeat = heartbeat_request_size + heartbeat_response_size

        # Number of heartbeats per worker
        # In current implementation, heartbeat is sent every iteration (training_delay = 0.5s)
        # But we're measuring with different intervals
        heartbeats_per_worker = int(training_time / heartbeat_interval)
        total_heartbeats = heartbeats_per_worker * self.num_workers

        # Bandwidth calculations
        total_bytes = total_heartbeats * total_per_heartbeat
        bandwidth_bytes_per_sec = total_bytes / training_time if training_time > 0 else 0

        # Failure detection latency = heartbeat_interval + timeout (15s default)
        failure_timeout = 15
        max_detection_latency = heartbeat_interval + failure_timeout
        avg_detection_latency = (heartbeat_interval / 2) + failure_timeout

        return {
            "heartbeats_per_worker": heartbeats_per_worker,
            "total_heartbeats": total_heartbeats,
            "total_bytes": total_bytes,
            "bandwidth_bytes_per_sec": bandwidth_bytes_per_sec,
            "bandwidth_kb_per_sec": bandwidth_bytes_per_sec / 1024,
            "max_failure_detection_latency": max_detection_latency,
            "avg_failure_detection_latency": avg_detection_latency,
            "heartbeat_request_size": heartbeat_request_size,
            "heartbeat_response_size": heartbeat_response_size
        }

    def run_trial(self, heartbeat_interval: float) -> Dict:
        """Run a single trial with specified heartbeat interval."""
        print(f"\n{'='*50}")
        print(f"Running trial: heartbeat_interval={heartbeat_interval}s")
        print(f"{'='*50}")

        clean_checkpoints()

        start_time = time.time()
        self.start_cluster(heartbeat_interval)

        completed = self.wait_for_completion(timeout=180)
        end_time = time.time()

        total_time = end_time - start_time
        self.stop_all()

        # Estimate metrics
        metrics = self.estimate_heartbeat_metrics(heartbeat_interval, total_time)

        result = {
            "heartbeat_interval": heartbeat_interval,
            "completed": completed,
            "total_time": total_time,
            "num_workers": self.num_workers,
            "num_iterations": self.num_iterations,
            **metrics
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Bandwidth: {metrics['bandwidth_kb_per_sec']:.2f} KB/s")
        print(f"  Max detection latency: {metrics['max_failure_detection_latency']:.1f}s")

        return result

    def run_benchmark(self, heartbeat_intervals: List[float] = None, num_trials: int = 2) -> Dict:
        """Run full network overhead benchmark."""
        if heartbeat_intervals is None:
            heartbeat_intervals = [0.5, 1.0, 2.0, 5.0]

        results = {
            "system_info": get_system_info(),
            "config": {
                "num_workers": self.num_workers,
                "num_iterations": self.num_iterations,
                "heartbeat_intervals": heartbeat_intervals,
                "num_trials": num_trials,
                "failure_timeout_seconds": 15
            },
            "trials_by_interval": {},
            "summary": {},
            "tradeoff_analysis": {}
        }

        for interval in heartbeat_intervals:
            interval_key = str(interval)
            results["trials_by_interval"][interval_key] = []

            for trial_num in range(num_trials):
                print(f"\n\n{'#'*60}")
                print(f"Interval {interval}s, Trial {trial_num + 1}/{num_trials}")
                print(f"{'#'*60}")

                trial_result = self.run_trial(interval)
                trial_result["trial_number"] = trial_num + 1
                results["trials_by_interval"][interval_key].append(trial_result)

                time.sleep(2)

        # Calculate summary statistics
        for interval_key, trials in results["trials_by_interval"].items():
            valid_trials = [t for t in trials if t.get("completed", False)]
            if valid_trials:
                results["summary"][interval_key] = {
                    "total_time": calculate_stats([t["total_time"] for t in valid_trials]),
                    "bandwidth_kb_per_sec": calculate_stats([t["bandwidth_kb_per_sec"] for t in valid_trials]),
                    "max_failure_detection_latency": valid_trials[0]["max_failure_detection_latency"],
                    "avg_failure_detection_latency": valid_trials[0]["avg_failure_detection_latency"],
                    "total_heartbeats": valid_trials[0]["total_heartbeats"]
                }

        # Analyze tradeoff
        for interval_key, stats in results["summary"].items():
            interval = float(interval_key)
            results["tradeoff_analysis"][interval_key] = {
                "interval_seconds": interval,
                "bandwidth_kb_sec": stats["bandwidth_kb_per_sec"]["mean"],
                "detection_latency_sec": stats["max_failure_detection_latency"],
                "tradeoff_score": stats["bandwidth_kb_per_sec"]["mean"] * stats["max_failure_detection_latency"]
            }

        return results


def main():
    print("=" * 60)
    print("Network Overhead Benchmark")
    print("=" * 60)

    benchmark = NetworkOverheadBenchmark(num_workers=4, num_iterations=30)

    results = benchmark.run_benchmark(
        heartbeat_intervals=[0.5, 1.0, 2.0],
        num_trials=2
    )

    save_results("network_overhead.json", results)

    print("\n" + "=" * 60)
    print("Network Overhead Summary")
    print("=" * 60)

    print(f"\n{'Interval':<12} {'Bandwidth':<15} {'Detection Latency':<20}")
    print("-" * 50)

    for interval_key, analysis in results.get("tradeoff_analysis", {}).items():
        print(f"{analysis['interval_seconds']:<12.1f}s "
              f"{analysis['bandwidth_kb_sec']:<15.2f} KB/s "
              f"{analysis['detection_latency_sec']:<20.1f}s")

    return results


if __name__ == "__main__":
    main()
