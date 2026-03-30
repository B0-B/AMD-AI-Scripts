#!/usr/bin/env python3
#
# Integrated benchmark pipeline for n-dimensional tests with multiple models,
# concurrencies, input- and output lengths and different datasets.
# Results will be casted to a .csv file once the benchmark completes.

# =========================== Benchmark Parameter ===========================
hf_models = ["amd/gpt-oss-120b-w-mxfp4-a-fp8",
             "amd/Llama-3.3-70B-Instruct-FP8-KV",
             "unsloth/Mistral-Small-3.2-24B-Instruct-2506-FP8",
             "RedHatAI/gemma-3-27b-it-FP8-dynamic",
             "amd/Llama-3.1-8B-Instruct-FP8-KV",
             "Qwen/Qwen3-4B-FP8",
             "Qwen/Qwen3-1.7B-FP8"]
hf_token = "<YOUR_HF-TOKEN>"
input_lengths = [1024, 4096]
output_lengths = [1024, 4096]
dataset_type = "random"
num_tokens = 50000
concurrencies = [1, 4, 16, 32, 64, 128]
device_type = "GPU"
device_name = "MI300X"
csv_path = f"./amd_{device_name.lower()}__vllm_integrated_benchmark_results.csv"
docker_image = "<YOUR_DOCKER_IMAGE_NAME>"
warmup_runs = 5
# ===========================================================================

# Run benchmark
import os
from vbench import integratedBenchmark

if __name__ == '__main__':

    if device_type.lower() == 'cpu':
        # Forces the runtime to see zero GPUs
        os.environ["ROCM_VISIBLE_DEVICES"] = "-1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # Optional: tell vLLM specifically to use the CPU backend
        os.environ["VLLM_TARGET_DEVICE"] = "cpu"

    print(f"[vBench]   Start integrated benchmark of {device_name} ...")
    integratedBenchmark(device_type,
                        device_name,
                        hf_models, 
                        hf_token, 
                        concurrencies, 
                        num_tokens, 
                        dataset_type, 
                        input_lengths, 
                        output_lengths, 
                        warmup_runs,
                        csv_output_path=csv_path,
                        docker_image=docker_image)