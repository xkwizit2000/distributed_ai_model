#!/bin/bash
# Gemma Distributed Inference Launch Script
#
# This script wraps PyTorch's torch.distributed.run module for distributed inference.
# It should be run from within the container on each node.

# Set ROCm environment variables to help with memory allocation issues
export HSA_XNACK=1
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
#
# Environment variables required:
#   NODE_RANK    - Rank of this node (0 for master, 1+ for workers)
#   MASTER_ADDR  - IP address of the master node
#   WORLD_SIZE   - Total number of nodes
#   MASTER_PORT  - Port for distributed communication (default: 29500)

set -e

# Defaults
MASTER_PORT=${MASTER_PORT:29500}
NNODES=${WORLD_SIZE:1}
NPROC_PER_NODE=1  # 1 process per node (using 1 GPU per node)

# Validate required variables
if [ -z "$NODE_RANK" ]; then
    echo "ERROR: NODE_RANK environment variable not set"
    exit 1
fi

if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: MASTER_ADDR environment variable not set"
    exit 1
fi

if [ -z "$WORLD_SIZE" ]; then
    echo "ERROR: WORLD_SIZE environment variable not set"
    exit 1
fi

echo "=== Gemma Distributed Inference Launch ==="
echo "Node Rank: $NODE_RANK"
echo "Master Address: $MASTER_ADDR:$MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Processes per Node: $NPROC_PER_NODE"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo ""

# Print ROCm device information
echo "=== ROCm Device Information ==="
rocm-smi || echo "rocm-smi not available"
echo ""

# For single-node testing, run the Python script directly
if [ "${NNODES:-1}" -eq 1 ] && [ "${NODE_RANK:-0}" -eq 0 ]; then
    echo "Running single-node inference..."
    python /app/scripts/run_gemma_inference.py
else
    # Launch using torch.distributed.run as an alternative to DeepSpeed
    echo "Running distributed inference with torch.distributed.run..."
    python -m torch.distributed.run --nnodes=${NNODES:-1} --nproc-per-node=${NPROC_PER_NODE:-1} \
        --node-rank=${NODE_RANK:-0} --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
        /app/scripts/run_gemma_inference.py
fi

echo "=== Inference Complete ==="