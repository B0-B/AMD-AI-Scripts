# AMD LLM Benchmark with vLLM

A benchmark CLI tool to perform benchmarks on any `CasualLM` model loaded with [vLLM](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/advanced/vllm/vllm.html) module to leverage the built-in ``flash-attention`` for acceleration.

## vBench

An integrated vLLM benchmark API.

## Features
- Supports CPU and GPU benchmarking
- Single benchmarks and whole test plans
- Multiple models, and sampling parameters
- Output test matrix as CSV table
- Accessible as module and CLI

## Requirements

- Linux (preferably Debian/Ubuntu 24.04)
- vLLM (preferably >=0.9.0)
- ROCm Driver (preferably >=6.4)
- Sufficient GPU memory depending on the model used
- Hugging Face account and API token (optional for locked models)

## Usage

### Single Configuration Benchmark
---

Run a single quick benchmark with fixed parameters. Use the CLI to tailor the benchmark to your needs.

```bash
python3 quick_bench.py \
    --hf_token hf_xxxx \
    --model inceptionai/jais-13b-chat \
    --num_tokens 10000 \
    --max_output_len 1024 \
    --max_input_len 512 \
    --dataset random \
    --temperature 0.1 \
    --batch_size 1
```

### Integrated Benchmark
---

This method can either be imported via `from vbench import integratedBenchmark` or run the prepared orchestration script `broad_bench.py`. In the ladder case we recommend to copy the script to not break the original template

```bash
cp broad_bench.py my_broad_bench.py
```

open the new ``my_broad_bench.py`` and adjust the benchmark parameter inside to fit your needs (see below)

```python
# =========================== Benchmark Parameter ===========================
hf_models = ["amd/gpt-oss-120b-w-mxfp4-a-fp8",
             "amd/Llama-3.3-70B-Instruct-FP8-KV",
             "unsloth/Mistral-Small-3.2-24B-Instruct-2506-FP8",
             "RedHatAI/gemma-3-27b-it-FP8-dynamic",
             "amd/Llama-3.1-8B-Instruct-FP8-KV",
             "Qwen/Qwen3-4B-FP8",
             "Qwen/Qwen3-1.7B-FP8"]
hf_token = None
input_lengths = [1024, 4096]
output_lengths = [1024, 4096]
dataset_type = "random"
num_tokens = 50000
concurrencies = [1, 4, 16, 32, 64, 128]
device_type = "GPU"
device_name = "MI300X"
csv_path = f"./amd_{device_name.lower()}__vllm_integrated_benchmark_results.csv"
docker_image = "my_docker_image_name"
warmup_runs = 5
# ===========================================================================
```

This script will create a whole test pipeline by iterating over all parameter variations and the final test matrix will be exported to the provided path as a ``.csv`` file.