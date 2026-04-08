# AMD LLM Benchmark with vLLM

The built-in benchmark shipped with the vLLM engine requires two separate servers by design - often separated in two containers, one for hosting the model and another for running the benchmark and prompting against the end-point. This approach is not only impractical, but may introduce query latency and overhead due to the TCP route between the containers. 
Instead, a simple customizable benchmark API can benchmark `CasualLM` models E2E on [vLLM](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/advanced/vllm/vllm.html) within a single thread/container.

## vBench

An integrated vLLM benchmark API with pre-tuned environment for AMD/ROCm.

## Features
- Supports CPU and GPU benchmarking
- Single benchmarks and whole test plans
- Multiple models, and sampling parameters
- Output test matrix as CSV file
- Accessible as module and CLI
- Laid-back ``docker-compose`` pipeline

## Requirements



| Requirement | Preferred Specification |
| :--- | :--- |
| **Operating System** | Linux (Debian/Ubuntu 24.04 recommended) |
| **vLLM Version** | >= 0.9.0 |
| **ROCm Driver** | >= 6.4 |
| **GPU Memory** | Sufficient VRAM based on model size |
| **Hugging Face** | Account & API Token (optional for locked models) |



## Usage

It is recommended to clone the entire project on the host system and change into the ``../AMD_LLM_vLLM_Benchmark`` directory.


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

<br>

### Integrated Benchmark
---

This method can either be imported via `from vbench import integratedBenchmark` or run the prepared orchestration script `broad_bench.py`. 

All benchmark arguments, parameters and environment settings are located in the ``docker-compose.yml``, please adjust the section below to fit your needs (see below).

```bash
nano docker-compose.yml
```

```yaml
# adjust the parameters for your benchmark in the section below.
...
# ==== Benchmark Arguments (will be ignored by docker) ====
    # ...
    x-benchmark-args:
      models:                                                       # Define the list of models to test
        - "amd/Llama-3.1-8B-Instruct-FP8-KV"
        - "inceptionai/jais-13b-chat"
      hf_token: "<hf_token_here>"                                   # Provide huggingface token
      device_type: GPU                                              # Which device type GPU or CPU
      device_name: MI300X                                           # Device name
      concurrencies: [1, 4]                                         # Number of parallel requests per iteration
      input_lengths: [512, 1024]                                    # Input lengths to iterate over
      output_lengths: [512, 1024]                                   # Output lengths to iterate over
      dataset_type: random                                          # Dataset "random" or "small-mixed"
      num_tokens: 10000                                             # Number of tokens to generate for 
                                                                    # each configuration (not needed if num_iterations is used).
      num_iterations: null                                          # Number of iterations or propagations per configuration
                                                                    # (not needed if num_tokens is used).
      csv_path: "./amd_vllm_integrated_benchmark_results.csv"       # CSV output path
      warmup_runs: 5                                                # Warmup iterations before each configuration
    # =========================================================
...
```

The `broad_bench.py` will automatically source these arguments and create a whole test pipeline by iterating over all test cases and the final test matrix will be exported to the provided path in ``.csv`` format.

<br>

### Broad Benchmark Launch Options
---


#### Direct Python Execution
---

The benchmark script can be executed in any prepared environment with a python interpreter (3.12 or later). Make sure that all necessary environment variables are set correctly. Optionally use the provided environment files

```bash
# Option A: Use your custom environment variables 
export HIP_VISIBLE_DEVICES=0
export ...

# Option B: Source the provided environment file (recommended)
set -a && source amd.env && set +a
```

The benchmark can be started with

```bash
python my_broad_bench.py
```

The benchmark results and process logs will be available in the `/results` sub-directory.


#### Docker Orchestration Pipeline
---

For docker setups this project provides a pre-configured test pipeline. Use the docker-compose.yml to adjust the orchestration to your needs, by default the orchestration will use ``amd.env`` tunable which will be sourced into the docker runtime, otherwise the paths can be adjusted to any custom tunable via the arguments in the ``docker-compose.yml`` - make sure the yaml is set to your needs.
The benchmark can be started with the provided ``run_broad_bench_docker.sh`` file which is verbose, or simply by using the raw docker compose command.

```bash
# Recommended
bash run_broad_bench_docker.sh

# Alternatively, use the raw docker command
docker compose run --rm vbench
```

After the benchmark the container will clean itself up, leaving no garbage behind. As before, all benchmark results and process logs will be available in the `/results` sub-directory. 

The progress can be inspected from the host system by simply inspecting the ``benchmark.log``

```bash
tail -n 20 results/benchmark.log
```

<br>


### Python Module & API Docs
---

The provided python module `vbench.py` provides the benchmarking tools as customizable python object.


#### `BaseBench` Class Documentation

The `BaseBench` class is a benchmarking utility designed to interface with the **vLLM** engine. It provides a standardized environment for loading Hugging Face models, managing prompt datasets, and measuring inference performance metrics such as Time-To-First-Token (TTFT).

#### Class: `BaseBench`

##### `__init__(self, hf_model, hf_token=None, device_type="GPU")`
Initializes the benchmarking environment and the underlying vLLM engine.

*   **Parameters:**
    *   `hf_model` (str): The Hugging Face model ID or a local path to the model.
    *   `hf_token` (str, optional): A Hugging Face authentication token for accessing gated models.
    *   `device_type` (str): The hardware target (e.g., "GPU", "CPU"). Defaults to "GPU".
*   **Key Attributes:**
    *   `self.llm`: The initialized vLLM `LLM` instance.
    *   `self.tokenizer`: The model's tokenizer used for encoding/decoding.
    *   `self.vocab_size`: The total number of tokens in the model's vocabulary.
    *   `self.dtype`: The detected precision/data type of the model weights.

---

##### `loadDataset(self, dataset_type)`
Configures the source for prompt generation.

*   **Parameters:**
    *   `dataset_type` (str): 
        *   `"random"`: Configures the class to generate synthetic prompts using random tokens.
        *   `"small-mixed"`: Loads real-world prompts from a local YAML file named `prompt_dataset__small_mixed.yml`.
*   **Behavior:** If `"small-mixed"` is selected, it attempts to resolve the file path relative to the script's location and parses the `prompts` key from the YAML content.

---

##### `prompt(self, prompts, sampling_params)`
Executes the inference request through the vLLM engine.

*   **Parameters:**
    *   `prompts` (list[str]): A list of string prompts to be processed.
    *   `sampling_params` (SamplingParams): A vLLM `SamplingParams` object defining generation settings (e.g., temperature, max_tokens).
*   **Returns:** A list of `RequestOutput` objects from vLLM.

---

##### `samplePromptVector(self, batch_size=1, random_input_length=None)`
Generates a batch of prompt strings based on the currently loaded dataset type.

*   **Parameters:**
    *   `batch_size` (int): Number of prompts to generate. Defaults to `1`.
    *   `random_input_length` (int, optional): Fixed token count for synthetic prompts. Defaults to `256` if using the "random" dataset.
*   **Returns:** `list[str]`: A list containing the generated or sampled prompt strings.

---

##### `ttft(self, sampling_params, batch_size=1, random_input_length=None, probings=10)`
Measures the **Time-To-First-Token** (TTFT) latency, representing the delay between request submission and the generation of the first token.

*   **Parameters:**
    *   `sampling_params` (SamplingParams): Configuration for the generation (recommend setting `max_tokens=1` for pure TTFT).
    *   `batch_size` (int): Number of concurrent prompts in the batch.
    *   `random_input_length` (int, optional): Token count for input prompts (relevant for prefill time).
    *   `probings` (int): Number of iterations to perform to gather statistical data.
*   **Returns:** `list[float, float, float]`:
    1.  **Mean**: The average TTFT in seconds.
    2.  **Median**: The middle value of the sorted results (resistant to outliers).
    3.  **Standard Deviation ($\sigma$)**: The statistical dispersion of the measurement.

---

#### Dependencies
To use this class, the following libraries must be available in the environment:
- `vllm`
- `pyyaml`
- `huggingface_hub`
- `pathlib`