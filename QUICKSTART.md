# Quick Start Guide - Distributed AI Model Inference

This guide walks you through setting up and running the distributed inference for the Gemma 4 26B model on your two Beelink devices.

## Prerequisites

1. **ROCm Drivers**: Ensure ROCm 6.x is installed on both host machines.
2. **Docker**: Installed and configured to allow GPU access.
3. **Network**: Both Beelinks must be on the same network and able to communicate.
4. **Hugging Face Account**: Required to access Gemma 4 26B model.
5. **OS version**: Tested using Ubuntu 26.04 LTS Minimal Server
6. **Packages**: 

## Step 1: Download Model Weights

Before building the container, you'll need to download the Gemma 4 26B model weights.

1. **Request Access**: Go to https://huggingface.co/google/gemma-4-26b-it and request access to the model repository.

2. **Install Hugging Face CLI**:
   ```bash
   pip install huggingface_hub
   ```

3. **Log in to Hugging Face**:
   ```bash
   huggingface-cli login
   ```
   Use your Hugging Face access token when prompted.

4. **Download the Model**:
   ```bash
   huggingface-cli download google/gemma-4-26b-it --local-dir /path/to/model_weights
   ```

   For better VRAM utilization, consider looking for a community quantized version of the model on Hugging Face.

   Store the model weights in a location accessible to both nodes (either on shared storage or downloaded separately on each node).

## Step 2: Build the Container Image (On Both Nodes) 

## Step 1: Build the Container Image (On Both Nodes)

The Docker image has a content size of approximately 10.3GB due to the base ROCm PyTorch image and GPU ML dependencies. This is normal for GPU-accelerated ML workloads.

```bash
cd /path/to/distributed_ai_model
docker build -t distributed_ai_model:latest .
```

After building, you can check the image size with:
```bash
docker images distributed_ai_model:latest
```

Note: The "disk usage" may appear much larger (around 40GB) due to filesystem overhead, but the "content size" (around 10.3GB) is what matters for storage and transfer.

## Step 3: Run on Node 0 (Master)

Identify the IP address of Node 0 (e.g., `192.168.1.100`).

```bash
docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host \
  -e NODE_RANK=0 \
  -e MASTER_ADDR=192.168.1.100 \
  -e MASTER_PORT=29500 \
  -e WORLD_SIZE=2 \
  -v /path/to/model_weights:/weights \
  distributed_ai_model:latest \
  /app/scripts/run_inference.sh
```

## Step 4: Run on Node 1 (Worker)

Use the same MASTER_ADDR as Node 0's IP.

```bash
docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host \
  -e NODE_RANK=1 \
  -e MASTER_ADDR=192.168.1.100 \
  -e MASTER_PORT=29500 \
  -e WORLD_SIZE=2 \
  -v /path/to/model_weights:/weights \
  distributed_ai_model:latest \
  /app/scripts/run_inference.sh
```

## Step 5: Verify and Test

- Both containers should start and initialize the distributed process.
- The model should load across both GPUs.
- Test inference by sending prompts to the running service.

## Troubleshooting

### Large Image Size
The Docker image has a content size of approximately 10.3GB, which is normal for GPU-accelerated ML workloads with ROCm and PyTorch. The base ROCm PyTorch image itself is around 39GB, so this size is expected.

For deployment to fixed nodes, this size is manageable since the image doesn't need to be transferred frequently.

### GPU Not Detected in Container
Ensure you're using the correct ROCm flags:
```bash
--device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host
```

### Network Communication Failed
- Check firewall settings on both nodes.
- Ensure port 29500 is open.
- Verify MASTER_ADDR is correct and reachable.

### Model Loading Errors
- Confirm model weights are present in the mounted volume.
- Check VRAM usage; 4-bit quantization is required for 26B models on 16GB GPUs.

## Next Steps: Kubernetes Deployment

Once you've validated the Docker setup, migrate to Kubernetes:

1. **Create ConfigMap**:
   ```bash
   kubectl apply -f k8s/configmap.yaml
   ```

2. **Create Headless Service**:
   ```bash
   kubectl apply -f k8s/service.yaml
   ```

3. **Deploy Job**:
   ```bash
   kubectl apply -f k8s/job.yaml
   ```

4. **Monitor**:
   ```bash
   kubectl get jobs
   kubectl get pods -l app=gemma-distributed
   ```

## Notes

- The current setup uses ZeRO-3 sharding to distribute the model.
- 4-bit quantization is essential for fitting 26B models into 2x 16GB VRAM.
- For production, consider adding health checks and persistent logging.
