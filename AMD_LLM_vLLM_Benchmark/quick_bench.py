#!/usr/bin/env python3
#
# vLLM Benchmark Script for LLM Inference
#
# Usage:
# python3 llm_vllm_benchmark.py \
#   --hf_token hf_xxxx \
#   --model inceptionai/jais-13b-chat \
#   --iterations 10 \
#   --warmup_runs 3 \
#   --max_input_len 1000 \
#   --max_output_len 1000 \
#   --batch_size 1

import argparse
from vbench import BaseBench, singleBenchmark, show

def main ():

    # Load Parser
    parser = argparse.ArgumentParser(description="LLM Benchmark Script For vLLM")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--model", type=str, required=True, help="Model huggingface path, e.g., inceptionai/jais-13b-chat")
    parser.add_argument("--num_iterations", type=int, default=10, help="Total number of batched propagations to run per configuration.")
    parser.add_argument("--dataset", type=int, default=10, help="random or small-mixed")
    parser.add_argument("--warmup_runs", type=int, default=3, help="How many warmup iterations.")
    parser.add_argument("--max_output_len", type=int, default=256, help="Max tokens to generate per request.")
    parser.add_argument("--max_input_len", type=int, default=256, help="Max tokens to input per request. Note: this parameter only works with random dataset.")
    parser.add_argument("--batch_size", type=int, default=1, help="The number of prompts processed concurrently.")
    parser.add_argument("--temperature", type=float, default=0, help="The sampling temperature, default=0")
    parser.add_argument("--top_p", type=float, default=1.0, help="Controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.")
    parser.add_argument("--device_type", type=str, default="GPU", help="Select device type GPU or CPU. Default: GPU")
    args = parser.parse_args()

    # Perform benchmark
    bench = BaseBench(args.model, args.hf_token)
    results = singleBenchmark(bench, 
                              args.batch_size,
                              args.num_iterations,
                              args.dataset, 
                              args.max_input_len, 
                              args.max_output_len, 
                              args.warmup_runs,
                              args.temperature,
                              args.top_p,
                              args.device_type)
    
    # Output
    show(results)