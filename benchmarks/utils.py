"""Utility functions for benchmarking distributed training orchestrator."""

import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
VIZ_DIR = PROJECT_ROOT / "benchmarks" / "visualizations"
MODELS_DIR = PROJECT_ROOT / "benchmarks" / "models"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ProcessManager:
    """Manages master and worker processes for benchmarking."""

    def __init__(self, num_workers: int = 4, checkpoint_interval: int = 10):
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        self.master_process: Optional[subprocess.Popen] = None
        self.worker_processes: List[subprocess.Popen] = []
        self.master_pid: Optional[int] = None
        self.worker_pids: List[int] = []

    def start_master(self, extra_args: List[str] = None) -> int:
        """Start the master process and return its PID."""
        cmd = [sys.executable, str(SRC_DIR / "master.py")]
        if extra_args:
            cmd.extend(extra_args)

        self.master_process = subprocess.Popen(
            cmd,
            cwd=str(SRC_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        self.master_pid = self.master_process.pid
        time.sleep(2)  # Wait for master to start
        return self.master_pid

    def start_workers(self, num_workers: Optional[int] = None) -> List[int]:
        """Start worker processes and return their PIDs."""
        num = num_workers or self.num_workers
        self.worker_processes = []
        self.worker_pids = []

        for _ in range(num):
            proc = subprocess.Popen(
                [sys.executable, str(SRC_DIR / "worker.py")],
                cwd=str(SRC_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            self.worker_processes.append(proc)
            self.worker_pids.append(proc.pid)
            time.sleep(0.5)

        return self.worker_pids

    def kill_worker(self, index: int = 0) -> Tuple[int, float]:
        """Kill a specific worker and return (PID, timestamp)."""
        if index >= len(self.worker_processes):
            raise ValueError(f"Worker index {index} out of range")

        pid = self.worker_pids[index]
        kill_time = time.time()
        os.kill(pid, signal.SIGKILL)
        return pid, kill_time

    def kill_all_workers(self) -> float:
        """Kill all workers and return timestamp."""
        kill_time = time.time()
        for proc in self.worker_processes:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        self.worker_processes = []
        self.worker_pids = []
        return kill_time

    def stop_all(self):
        """Stop all processes gracefully."""
        for proc in self.worker_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

        if self.master_process:
            try:
                self.master_process.terminate()
                self.master_process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    self.master_process.kill()
                except ProcessLookupError:
                    pass

        self.worker_processes = []
        self.worker_pids = []
        self.master_process = None
        self.master_pid = None

    def is_master_running(self) -> bool:
        """Check if master is still running."""
        if self.master_process is None:
            return False
        return self.master_process.poll() is None

    def wait_for_completion(self, timeout: float = 300) -> bool:
        """Wait for all processes to complete."""
        start = time.time()
        while time.time() - start < timeout:
            all_done = all(p.poll() is not None for p in self.worker_processes)
            if all_done:
                return True
            time.sleep(1)
        return False


class BenchmarkTimer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.duration = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


def save_results(filename: str, data: Dict):
    """Save benchmark results to JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to: {filepath}")


def load_results(filename: str) -> Dict:
    """Load benchmark results from JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'r') as f:
        return json.load(f)


def get_system_info() -> Dict:
    """Get system information for reproducibility."""
    import platform

    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def clean_checkpoints():
    """Clean up checkpoint directories."""
    checkpoint_dir = Path("/tmp/checkpoints")
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)

    state_file = Path("/tmp/orchestrator_state.json")
    if state_file.exists():
        state_file.unlink()


def calculate_stats(values: List[float]) -> Dict:
    """Calculate statistics for a list of values."""
    import statistics
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "count": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "count": len(values)
    }
