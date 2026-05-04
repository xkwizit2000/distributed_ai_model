# Distributed AI Model Inference Container
# Optimized build to reduce image size

FROM docker.io/rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.10.0

# Set working directory
WORKDIR /app

# Install Python dependencies for DeepSpeed and Hugging Face
# Using --no-cache-dir to avoid storing pip cache
RUN pip install --no-cache-dir \
    deepspeed \
    transformers \
    accelerate \
    bitsandbytes \
    safetensors \
    huggingface_hub \
    torchrun && \
    # Clean up pip cache
    rm -rf /root/.cache/pip/*

# Clean up unnecessary packages/files
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Remove documentation files
    find /usr/share/doc -type f -delete && \
    find /opt/venv/lib/python3.10/site-packages -name "*.pyc" -delete && \
    find /opt/venv/lib/python3.10/site-packages -name "__pycache__" -type d -exec rm -rf {} + && \
    # Remove tests directories if they exist
    find /opt/venv/lib/python3.10/site-packages -name "tests" -type d -exec rm -rf {} + && \
    find /opt/venv/lib/python3.10/site-packages -name "test" -type d -exec rm -rf {} + && \
    # Remove examples directories if they exist
    find /opt/venv/lib/python3.10/site-packages -name "examples" -type d -exec rm -rf {} +

# Copy DeepSpeed configuration
COPY deepspeed_config.json /app/deepspeed_config.json

# Copy inference scripts
COPY scripts/ /app/scripts/

# Set environment variables for ROCm
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV PYTORCH_CUDA_ALLOC_CONF=allow_all
ENV HSA_XNACK=1
ENV HIP_VISIBLE_DEVICES=0
ENV ROCR_VISIBLE_DEVICES=0
ENV PYTORCH_HIP_ALLOC_CONF=backend:cudaMallocAsync

# Default command (can be overridden by docker run)
# The actual launch command will be provided at runtime
CMD ["python", "-c", "print('Container ready. Use torchrun or deepspeed to launch.')"]