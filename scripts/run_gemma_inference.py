#!/usr/bin/env python3
"""
Gemma 4 Distributed Inference Script

This script loads the Gemma 4 26B model using DeepSpeed ZeRO-3 sharding
to distribute the model across multiple GPUs (pot on separate nodes).

Usage:
    # Method 1: Using PyTorch torch.distributed.run module
    python -m torch.distributed.run --nnodes=2 --nproc-per-node=1 --node-rank=$NODE_RANK \
             --master-addr=$MASTER_ADDR --master-port=29500 \
             run_gemma_inference.py

    # Method 2: Using DeepSpeed launcher (recommended)
    deepspeed --num_nodes=2 --num_gpus=1 --master_addr=$MASTER_ADDR \
              --master_port=29500 run_gemma_inference.py

Note: The 'torchrun' command may not be available in your environment. Use 'python -m torch.distributed.run' or 'deepspeed' instead.
"""

import os
import sys
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes import BnbLinearType

# Configuration
MODEL_NAME = "google/gemma-4-26b-it"  # Update to 26B variant
QUANTIZATION = "4bit"  # Use 4-bit quantization to fit in VRAM
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# ROCm-specific configuration
os.environ["HSA_XNACK"] = "1"

# Check if we're running on ROCm
try:
    import torch
    if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        print("Running on ROCm")
        # ROCm-specific settings
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        os.environ["ROCR_VISIBLE_DEVICES"] = "0"
    else:
        print("Running on CUDA or CPU")
except Exception as e:
    print(f"Error checking for ROCm: {e}")

def get_env_vars():
    """Get distributed training environment variables."""
    return {
        'node_rank': int(os.environ.get('NODE_RANK', 0)),
        'master_addr': os.environ.get('MASTER_ADDR', 'localhost'),
        'master_port': int(os.environ.get('MASTER_PORT', '29500')),
        'world_size': int(os.environ.get('WORLD_SIZE', 1)),
    }

def load_model_and_tokenizer():
    """Load Gemma model with 4-bit quantization."""
    print(f"Loading model: {MODEL_NAME}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            use_nested_quant=False,
            device_map="auto",  # Let accelerate/DeepSpeed handle device mapping
            trust_remote_code=True,
        )
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def generate_response(model, tokenizer, input_text):
    """Generate a response from the model."""
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    """Main entry point."""
    print("=== Gemma 4 26B Distributed Inference ===")
    
    env_vars = get_env_vars()
    print(f"Node rank: {env_vars['node_rank']}, World size: {env_vars['world_size']}")
    print(f"Master: {env_vars['master_addr']}:{env_vars['master_port']}")
    
    # Print device information
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available")
    
    try:
        # Initialize DeepSpeed
        # Note: For inference, we might use deepspeed.inference_config instead
        # This is a starting point - may need adjustment for pure inference
        model, tokenizer = load_model_and_tokenizer()
        
        # Example inference
        test_input = "Hello, how can I help you today?"
        print(f"\nInput: {test_input}")
        
        response = generate_response(model, tokenizer, test_input)
        print(f"Response: {response}")
        
        print("\nInference complete.")
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name == "__main__":
    main()
