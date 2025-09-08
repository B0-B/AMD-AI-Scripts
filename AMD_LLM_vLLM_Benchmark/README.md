# AMD LLM Benchmark with vLLM

A benchmark CLI tool to perform benchmarks on any `CasualLM` model loaded with [vLLM](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/advanced/vllm/vllm.html) module to leverage the built-in ``flash-attention`` for acceleration.

## Features
- Performs subsequent iterations of a given test configuration. 
- Builds random batch vectors to simulate real case scenario
- Allows to modify token size
- Calculates throughput in tokens and requests, latencies etc.

## Requirements

- Linux (preferably Debian/Ubuntu 24.04)
- vLLM (preferably >=0.9.0)
- ROCm Driver (preferably >=6.4)
- Sufficient GPU memory depending on the model used
- Hugging Face account and API token (optional for locked models)

## Usage

Run a benchmark with a single command line and tailor the benchmark to your needs.


```bash
python3 llm_vllm_benchmark.py \
    --hf_token hf_xxxx \
    --model inceptionai/jais-13b-chat \
    --iterations 10 \
    --warmup_iterations 3 \
    --max_new_tokens 200 \
    --batch_size 1
```