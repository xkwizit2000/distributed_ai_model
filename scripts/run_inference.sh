#!/bin/bash
# Gemma Distributed Inference Launch Script
#
# This script wraps PyTorch's torch.distributed.run module for distributed inference.
# It should be run from within the container on each node.
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
echo ""

# Launch using PyTorch's torch.distributed.run module
# Note: Use 'python -m torch.distributed.run' instead of 'torchrun' command
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc-per-node=$NPROC_PER_NODE \
    --node-rank=$NODE_RANK \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    /app/scripts/run_gemma_inference.py

echo "=== Inference Complete ==="