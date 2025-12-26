# Distributed Training Orchestrator

A distributed training orchestration system using gRPC and PyTorch DDP.

## Project Structure

```
distributed/
├── protos/
│   └── orchestrator.proto      # gRPC service definition
├── src/
│   ├── master.py               # Orchestrator server
│   ├── worker.py               # DDP training worker
│   ├── orchestrator_pb2.py     # Generated message classes
│   └── orchestrator_pb2_grpc.py # Generated service stubs
├── scripts/
│   ├── run.sh                  # Launch master + 4 workers
│   └── generate_proto.sh       # Regenerate proto files
├── requirements.txt
└── README.md
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Master (port 50051)                       │
│  - Worker registry & rank assignment                        │
│  - Training coordination                                    │
│  - Live dashboard                                           │
└─────────────────────────────────────────────────────────────┘
        ▲               ▲               ▲               ▲
        │ gRPC          │ gRPC          │ gRPC          │ gRPC
        ▼               ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Worker  │    │ Worker  │    │ Worker  │    │ Worker  │
   │ Rank 0  │    │ Rank 1  │    │ Rank 2  │    │ Rank 3  │
   └─────────┘    └─────────┘    └─────────┘    └─────────┘
        ▲               ▲               ▲               ▲
        └───────────────┴───────┬───────┴───────────────┘
                                │
                    PyTorch DDP (port 29500)
```

## Setup

1. Create conda environment:
```bash
conda create -n myproject python=3.10
conda activate myproject
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate proto files (if needed):
```bash
./scripts/generate_proto.sh
```

## Usage

### Run everything (master + 4 workers):
```bash
./scripts/run.sh
```

### Run manually:
```bash
# Terminal 1: Master
cd src && python master.py

# Terminals 2-5: Workers
cd src && python worker.py
```

## gRPC Services

| RPC | Description |
|-----|-------------|
| `Register` | Worker registers, receives rank |
| `SendHeartbeat` | Worker sends training progress |
| `CanStartTraining` | Check if all workers ready |

## Flow

1. Master starts on port 50051
2. Workers register (get ranks 0, 1, 2, 3)
3. Workers poll `CanStartTraining` until ready
4. All workers init DDP process group (gloo backend)
5. Training loop with heartbeats
6. Master displays live dashboard


