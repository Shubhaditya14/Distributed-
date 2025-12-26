from orchestrator_pb2_grpc import OrchestratorServicer
import orchestrator_pb2
import grpc
import orchestrator_pb2_grpc
from concurrent import futures
import time
import json
import threading

class OrchestratorServicerMaster(OrchestratorServicer):
    def __init__(self, expected_workers=4, checkpoint_interval=10):
        print(f"Master expecting {expected_workers} workers.")
        self.workers = {}
        self.next_rank = 0
        self.expected_workers = expected_workers
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_iteration = 0
        self.checkpoint_completion = {}  # Iteration -> set of worker_ids
        self.last_complete_checkpoint = 0
        self.in_recovery = False
        self.lock = threading.Lock()
        self.all_workers_ready = threading.Condition(self.lock)
    
    def Register(self, request, context):
        with self.lock:
            worker_id = request.worker_id

            if worker_id in self.workers:
                return orchestrator_pb2.RegisterResponse(
                    assigned_rank=self.workers[worker_id]['rank'],
                    world_size=len(self.workers),
                    success=False,
                    message=f"Worker {worker_id} already registered"
                )
            
            assigned_rank = self.next_rank
            self.next_rank += 1

            self.workers[worker_id] = {
                'rank': assigned_rank,
                'current_iteration': 0,
                'current_loss': 0.0,
                'is_training': False,
                'last_heartbeat': time.time(),
            }

            print(f"Worker {worker_id} registered with rank {assigned_rank}")

            if len(self.workers) == self.expected_workers:
                print(f"All {self.expected_workers} workers registered. Notifying all workers.")
                self.all_workers_ready.notify_all()

            return orchestrator_pb2.RegisterResponse(
                assigned_rank=assigned_rank,
                world_size=self.expected_workers,
                success=True,
                message="Registration successful"
            ) 
    
    def SendHeartbeat(self, request, context):
        with self.lock:
            worker_id = request.worker_id

            if worker_id in self.workers:
                self.workers[worker_id]['current_iteration'] = request.current_iteration
                self.workers[worker_id]['current_loss'] = request.current_loss
                self.workers[worker_id]['is_training'] = request.is_training
                self.workers[worker_id]['last_heartbeat'] = time.time()

            if request.checkpointed_iteration > 0:
                iteration = request.checkpointed_iteration
                if iteration not in self.checkpoint_completion:
                    self.checkpoint_completion[iteration] = set()
                self.checkpoint_completion[iteration].add(worker_id)
                
                if len(self.checkpoint_completion[iteration]) == self.expected_workers:
                    self.last_complete_checkpoint = iteration
                    print(f"*** All workers completed checkpoint for iteration {iteration} ***")
                    self.save_state_to_disk()

            if request.is_training:
                print(f"Worker {worker_id} - iteration: {request.current_iteration}, loss: {request.current_loss:.4f}")
            else:
                print(f"Worker {worker_id} sent heartbeat at timestamp {request.timestamp}")

            self.display_dashboard()

            should_checkpoint = self.check_checkpoint_condition()

            return orchestrator_pb2.HeartbeatResponse(
                acknowledged=True,
                should_checkpoint=should_checkpoint
            )

    def check_checkpoint_condition(self):
        if len(self.workers) < self.expected_workers:
            return False

        iterations = [w['current_iteration'] for w in self.workers.values()]
        min_iteration = min(iterations) if iterations else 0

        if min_iteration == 0:
            return False

        if (min_iteration % self.checkpoint_interval == 0 and
            min_iteration > self.last_checkpoint_iteration):
            self.last_checkpoint_iteration = min_iteration
            print(f"\n*** CHECKPOINT TRIGGERED at iteration {min_iteration} ***\n")
            return True

        return False

    def CanStartTraining(self, request, context):
        with self.lock:
            while len(self.workers) < self.expected_workers:
                print(f"Waiting for workers: {len(self.workers)}/{self.expected_workers}")
                self.all_workers_ready.wait()
            
            print(f"All {self.expected_workers} workers registered. Training can start!")
            return orchestrator_pb2.CanStartTrainingResponse(
                ready=True,
                world_size=self.expected_workers
            )
            
    def GetRecoveryInfo(self, request, context):
        with self.lock:
            return orchestrator_pb2.RecoveryResponse(
                in_recovery_mode=self.in_recovery,
                checkpoint_iteration=self.last_complete_checkpoint
            )

    def display_dashboard(self):
        print("\n" + "=" * 50)
        print("TRAINING DASHBOARD")
        print("=" * 50)
        for worker_id, info in self.workers.items():
            status = "TRAINING" if info['is_training'] else "IDLE"
            print(f"Rank {info['rank']} ({worker_id}): {status} | iter: {info['current_iteration']} | loss: {info['current_loss']:.4f}")
        print("=" * 50 + "\n")
        
    def save_state_to_disk(self):
        state = {
            'last_complete_checkpoint': self.last_complete_checkpoint
        }
        with open("/tmp/orchestrator_state.json", "w") as f:
            json.dump(state, f)
        print(f"Saved orchestrator state to /tmp/orchestrator_state.json")

    def trigger_recovery(self):
        with self.lock:
            if self.in_recovery:
                return
            
            print("!!! Failure Detected - Triggering Recovery !!!")
            self.in_recovery = True

            # Reset worker state to force re-registration
            self.workers = {}
            self.next_rank = 0
            
            print("Workers killed. Please run ./run.sh to restart")

    def failure_detector_thread(self):
        while True:
            time.sleep(5)
            with self.lock:
                current_time = time.time()
                for worker_id, info in list(self.workers.items()):
                    if current_time - info['last_heartbeat'] > 15: # 15 sec timeout
                        print(f"Worker {worker_id} failed!")
                        self.trigger_recovery()
                        break # Only trigger once per check
    
def serve():
    servicer = OrchestratorServicerMaster()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    orchestrator_pb2_grpc.add_OrchestratorServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Orchestrator server started on port 50051")

    # Start failure detector thread
    threading.Thread(target=servicer.failure_detector_thread, daemon=True).start()

    server.wait_for_termination()


if __name__ == '__main__':
    serve()
