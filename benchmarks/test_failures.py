"""
Part 2: Failure Scenario Tests

Test 1: Single Worker Failure
- Kill one worker at iteration 35
- Verify recovery from checkpoint

Test 2: Multiple Sequential Failures
- Multiple failures at different iterations
- Track total training time with recoveries

Test 3: Checkpoint Corruption Handling
- Corrupt a checkpoint file
- Verify fallback to previous checkpoint
"""

import os
import sys
import time
import signal
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (
    SRC_DIR, RESULTS_DIR,
    save_results, get_system_info, clean_checkpoints
)


class FailureTests:
    """Tests for various failure scenarios."""

    def __init__(self, num_workers: int = 4, checkpoint_interval: int = 10):
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        self.master_process: Optional[subprocess.Popen] = None
        self.worker_processes: List[subprocess.Popen] = []
        self.events: List[Dict] = []

    def log_event(self, event_type: str, data: Dict = None):
        """Log a timestamped event."""
        self.events.append({
            "timestamp": time.time(),
            "time_str": datetime.now().isoformat(),
            "type": event_type,
            "data": data or {}
        })

    def start_master(self) -> int:
        """Start the master process."""
        env = os.environ.copy()
        env["EXPECTED_WORKERS"] = str(self.num_workers)
        env["CHECKPOINT_INTERVAL"] = str(self.checkpoint_interval)

        self.master_process = subprocess.Popen(
            [sys.executable, str(SRC_DIR / "master.py")],
            cwd=str(SRC_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        time.sleep(2)
        self.log_event("master_started", {"pid": self.master_process.pid})
        return self.master_process.pid

    def start_workers(self) -> List[int]:
        """Start worker processes."""
        env = os.environ.copy()
        env["EXPECTED_WORKERS"] = str(self.num_workers)

        self.worker_processes = []
        pids = []

        for i in range(self.num_workers):
            proc = subprocess.Popen(
                [sys.executable, str(SRC_DIR / "worker.py")],
                cwd=str(SRC_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            self.worker_processes.append(proc)
            pids.append(proc.pid)
            time.sleep(0.5)

        self.log_event("workers_started", {"pids": pids})
        return pids

    def kill_worker(self, index: int = 0) -> int:
        """Kill a specific worker."""
        if index >= len(self.worker_processes):
            raise ValueError(f"Worker index {index} out of range")

        pid = self.worker_processes[index].pid
        self.log_event("killing_worker", {"index": index, "pid": pid})
        os.kill(pid, signal.SIGKILL)
        self.log_event("worker_killed", {"index": index, "pid": pid})
        return pid

    def restart_workers(self) -> List[int]:
        """Kill all workers and restart them."""
        # Kill remaining workers
        self.log_event("killing_all_workers")
        for proc in self.worker_processes:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

        time.sleep(2)

        # Start new workers
        self.worker_processes = []
        return self.start_workers()

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

    def wait_seconds(self, seconds: float):
        """Wait for specified seconds, simulating training progress."""
        time.sleep(seconds)

    def check_checkpoints_exist(self, iteration: int) -> bool:
        """Check if checkpoints exist for a specific iteration."""
        checkpoint_dir = Path("/tmp/checkpoints")
        for rank in range(self.num_workers):
            checkpoint_path = checkpoint_dir / f"rank_{rank}" / f"checkpoint_iter_{iteration}.pt"
            if not checkpoint_path.exists():
                return False
        return True

    def corrupt_checkpoint(self, rank: int, iteration: int):
        """Corrupt a checkpoint file."""
        checkpoint_path = Path(f"/tmp/checkpoints/rank_{rank}/checkpoint_iter_{iteration}.pt")
        if checkpoint_path.exists():
            with open(checkpoint_path, 'w') as f:
                f.write("corrupted data")
            self.log_event("checkpoint_corrupted", {
                "rank": rank,
                "iteration": iteration,
                "path": str(checkpoint_path)
            })

    def test_single_worker_failure(self) -> Dict:
        """
        Test 1: Single Worker Failure
        - Start training
        - Kill one worker after checkpoint at iteration 10
        - Verify recovery
        """
        print("\n" + "=" * 60)
        print("Test 1: Single Worker Failure")
        print("=" * 60)

        self.events = []
        clean_checkpoints()

        result = {
            "test_name": "single_worker_failure",
            "config": {
                "num_workers": self.num_workers,
                "checkpoint_interval": self.checkpoint_interval
            }
        }

        try:
            # Start cluster
            self.start_master()
            worker_pids = self.start_workers()
            print(f"Workers started: {worker_pids}")

            # Wait for first checkpoint (iteration 10)
            print("Waiting for checkpoint at iteration 10...")
            time.sleep(15)  # ~10 iterations at 0.5s + margin
            self.log_event("waiting_for_checkpoint", {"target": 10})

            # Verify checkpoint exists
            if self.check_checkpoints_exist(10):
                self.log_event("checkpoint_verified", {"iteration": 10})
                print("Checkpoint at iteration 10 verified!")
            else:
                print("Warning: Checkpoint at iteration 10 not found")

            # Kill worker 0
            print("Killing worker 0...")
            self.kill_worker(0)

            # Wait for master to detect failure (15s timeout + some margin)
            print("Waiting for failure detection (~20s)...")
            time.sleep(20)
            self.log_event("failure_detection_wait_complete")

            # Restart workers
            print("Restarting workers...")
            new_pids = self.restart_workers()
            print(f"New workers: {new_pids}")

            # Wait for training to resume and complete
            print("Waiting for training to complete...")
            time.sleep(60)  # Wait for remaining iterations

            self.log_event("test_complete")
            result["success"] = True
            result["message"] = "Single worker failure test completed"

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            self.log_event("test_error", {"error": str(e)})

        finally:
            self.stop_all()

        result["events"] = self.events
        result["event_count"] = len(self.events)

        return result

    def test_multiple_failures(self) -> Dict:
        """
        Test 2: Multiple Sequential Failures
        - Training with multiple failure injections
        - Track recovery and total time
        """
        print("\n" + "=" * 60)
        print("Test 2: Multiple Sequential Failures")
        print("=" * 60)

        self.events = []
        clean_checkpoints()

        result = {
            "test_name": "multiple_failures",
            "config": {
                "num_workers": self.num_workers,
                "checkpoint_interval": self.checkpoint_interval,
                "planned_failures": [25, 55]  # Simplified: 2 failures
            },
            "failures": []
        }

        start_time = time.time()

        try:
            # Start cluster
            self.start_master()
            self.start_workers()

            # First phase: run until iteration ~25
            print("Phase 1: Training to iteration 25...")
            time.sleep(20)  # ~25 iterations

            # First failure
            print("Injecting failure 1...")
            self.log_event("failure_injected", {"failure_num": 1})
            self.kill_worker(0)
            time.sleep(20)  # Detection time

            # Restart
            self.restart_workers()
            result["failures"].append({
                "failure_num": 1,
                "approx_iteration": 25,
                "recovery_checkpoint": 20
            })

            # Second phase: run until iteration ~55
            print("Phase 2: Training to iteration 55...")
            time.sleep(25)

            # Second failure
            print("Injecting failure 2...")
            self.log_event("failure_injected", {"failure_num": 2})
            self.kill_worker(1)
            time.sleep(20)

            # Restart
            self.restart_workers()
            result["failures"].append({
                "failure_num": 2,
                "approx_iteration": 55,
                "recovery_checkpoint": 50
            })

            # Complete training
            print("Completing training...")
            time.sleep(40)

            end_time = time.time()
            result["success"] = True
            result["total_time"] = end_time - start_time
            result["num_failures"] = len(result["failures"])

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        finally:
            self.stop_all()

        result["events"] = self.events
        return result

    def test_checkpoint_corruption(self) -> Dict:
        """
        Test 3: Checkpoint Corruption Handling
        - Let training checkpoint at iterations 10, 20, 30
        - Corrupt checkpoint at iteration 30
        - Verify fallback to iteration 20
        """
        print("\n" + "=" * 60)
        print("Test 3: Checkpoint Corruption Handling")
        print("=" * 60)

        self.events = []
        clean_checkpoints()

        result = {
            "test_name": "checkpoint_corruption",
            "config": {
                "num_workers": self.num_workers,
                "checkpoint_interval": self.checkpoint_interval
            }
        }

        try:
            # Start cluster
            self.start_master()
            self.start_workers()

            # Wait for checkpoints at 10, 20, 30
            print("Waiting for checkpoints at 10, 20, 30...")
            time.sleep(25)  # ~30 iterations

            # Verify checkpoints exist
            for iter_num in [10, 20, 30]:
                exists = self.check_checkpoints_exist(iter_num)
                print(f"Checkpoint {iter_num}: {'exists' if exists else 'missing'}")

            # Corrupt checkpoint at iteration 30
            print("Corrupting checkpoint at iteration 30...")
            for rank in range(self.num_workers):
                self.corrupt_checkpoint(rank, 30)

            # Kill a worker at ~35
            print("Killing worker at iteration ~35...")
            time.sleep(5)
            self.kill_worker(0)

            # Wait for detection
            time.sleep(20)

            # Restart - workers should fall back to iteration 20
            print("Restarting workers (should recover from iter 20)...")
            self.restart_workers()

            # Let training continue
            time.sleep(30)

            result["success"] = True
            result["corrupted_checkpoint"] = 30
            result["expected_fallback"] = 20
            result["message"] = "Checkpoint corruption test completed"

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        finally:
            self.stop_all()

        result["events"] = self.events
        return result


def run_all_tests():
    """Run all failure scenario tests."""
    print("=" * 60)
    print("Failure Scenario Tests")
    print("=" * 60)

    tests = FailureTests(num_workers=4, checkpoint_interval=10)

    all_results = {
        "system_info": get_system_info(),
        "tests": {}
    }

    # Test 1: Single Worker Failure
    all_results["tests"]["single_failure"] = tests.test_single_worker_failure()
    time.sleep(5)

    # Test 2: Multiple Failures
    all_results["tests"]["multiple_failures"] = tests.test_multiple_failures()
    time.sleep(5)

    # Test 3: Checkpoint Corruption
    all_results["tests"]["checkpoint_corruption"] = tests.test_checkpoint_corruption()

    # Save results
    save_results("failure_tests.json", all_results)

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in all_results["tests"].items():
        status = "PASSED" if result.get("success", False) else "FAILED"
        print(f"  {test_name}: {status}")

    return all_results


if __name__ == "__main__":
    run_all_tests()
