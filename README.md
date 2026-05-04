# Distributed AI Model Inference

Distributed inference setup for Gemma 4 26B model across multiple AMD GPU nodes using DeepSpeed and containerization.

## Docker Image Size

The Docker image has a content size of approximately 10.3GB. This is primarily due to the base ROCm PyTorch image (39GB) which includes the full ROCm runtime, PyTorch with GPU support, and other dependencies required for GPU-accelerated ML workloads. While we've implemented optimizations to clean up unnecessary files, the inherent size of the GPU ML stack makes this size reasonable for this type of workload.

For deployment to fixed nodes, this size is manageable and the image doesn't need to be transferred frequently.

## Architecture

- **Framework**: DeepSpeed with ZeRO-3 sharding
- **Model**: Gemma 4 26B or 31B (quantized 4-bit)
- **Hardware**: AMD 6800M GPUs (16GB VRAM each)
- **Containerization**: Docker (ROCm base image)
- **Future Path**: Kubernetes deployment

## Project Structure

```
distributed_ai_model/
├── README.md           # Project documentation
├── Dockerfile          # Multi-stage container build instructions
├── deepspeed_config.json # DeepSpeed ZeRO configuration
├── scripts/
│   ├── run_gemma_inference.py # Main inference script
│   ├── run_inference.sh       # Launch wrapper script
├── k8s/
│   ├── pod.yaml        # Kubernetes pod template (future)
│   ├── service.yaml    # Headless service for worker communication (future)
│   ├── job.yaml        # Kubernetes Job definition (future)
│   ├── configmap.yaml  # DeepSpeed config ConfigMap (future)
```

## Quick Start (Two Node Setup)

1. **Build the container image** on both nodes:
   ```bash
   docker build -t distributed_ai_model:latest .
   ```

2. **Run on Node 0 (Master)**:
   ```bash
   docker run --rm --network=host \
     --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host \
     -e NODE_RANK=0 -e MASTER_ADDR=192.168.1.100 -e WORLD_SIZE=2 \
     -v /path/to/model_weights:/weights \
     distributed_ai_model:latest
   ```

3. **Run on Node 1 (Worker)**:
   ```bash
   docker run --rm --network=host \
     --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host \
     -e NODE_RANK=1 -e MASTER_ADDR=192.168.1.100 -e WORLD_SIZE=2 \
     -v /path/to/model_weights:/weights \
     distributed_ai_model:latest
   ```

## Configuration

- `deepspeed_config.json`: Defines ZeRO-3 sharding, offloading, and communication settings.
- Environment Variables:
  - `NODE_RANK`: Rank of this node (0 for master, 1+ for workers)
  - `MASTER_ADDR`: IP address of the master node
  - `MASTER_PORT`: Port for distributed communication (default: 29500)
  - `WORLD_SIZE`: Total number of nodes

## Model Quantization

Gemma 4 26B/31B requires 4-bit quantization to fit within 2x 16GB VRAM. The inference script uses `bitsandbytes` for this purpose.

## Next Steps

1. Review and customize `deepspeed_config.json`
2. Adapt `run_gemma_inference.py` for your specific Gemma model variant
3. Build and test the container on a single node first
3. Test distributed launch across both nodes
4. (Future) migrate to Kubernetes deployment using configs in `k8s/`

## Notes

- Ensure both nodes can communicate over the network (no firewall blocking port 29500)
- ROCm drivers must be installed on the host for GPU access in containers
- Model weights should be stored on a shared volume or downloaded by each node
