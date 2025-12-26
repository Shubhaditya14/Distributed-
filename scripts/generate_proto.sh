#!/bin/bash

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda environment
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate myproject 2>/dev/null

echo "Generating Python code from proto files..."

python -m grpc_tools.protoc \
    -I"$PROJECT_ROOT/protos" \
    --python_out="$PROJECT_ROOT/src" \
    --grpc_python_out="$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/protos/orchestrator.proto"

echo "Done! Generated files in src/"
