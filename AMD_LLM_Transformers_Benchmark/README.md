# AMD LLM Transformers Benchmark

A benchmark CLI tool to perform benchmarks on any `CasualLM` model loaded with [transformers](https://github.com/huggingface/transformers) module.

## Features
- Performs subsequent iterations of a given test configuration. 
- Builds random batch vectors to simulate real case scenario
- Allows to modify token size
- Calculates throughput in tokens and requests, latencies etc.

## Requirements

- Linux (preferably Debian/Ubuntu 24.04)
- ROCm Driver (preferably >=6.4)
- Sufficient GPU memory depending on the model used
- Hugging Face account and API token (optional for locked models)

## Usage

Run a benchmark with a single command line and tailor the benchmark to your needs.


```bash
python3 llm_transformers_benchmark.py \
    --hf_token hf_xxxx \
    --model inceptionai/jais-13b-chat \
    --iterations 10 \
    --warmup_iterations 3 \
    --max_new_tokens 200 \
    --batch_size 1
```