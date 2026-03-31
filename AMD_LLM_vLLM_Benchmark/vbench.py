#!/usr/bin/env python3
import os, gc, sys
# Try importing yaml, otherwise install it on-the-fly
try:
    import yaml
except ImportError:
    # Use sys.executable to target the current python environment
    os.system(f"{sys.executable} -m pip install pyyaml")
    import yaml
import torch
from time import perf_counter
from pathlib import Path
from random import sample, randint
from huggingface_hub import login
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel


# =============== Base Bench Object =================
class BaseBench:

    def __init__(self, hf_model: str, hf_token: str|None=None, device_type: str="GPU"):
        
        self.location = Path(__file__).resolve().parent

        if hf_token:
            login(token=hf_token)

        self.hf_model = hf_model
        self.llm = LLM(
            model=hf_model,
            tokenizer=hf_model,
            trust_remote_code=True,
            tokenizer_mode="auto",
            download_dir=None,
            hf_token=hf_token,
            disable_log_stats = False, # enable metrics collection
            enforce_eager=True 
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.dtype = self.llm.llm_engine.model_config.dtype

        # Random dataset 
        self.dataset_type: str|None = None
        self.dataset: list[str]|None = None

        # Small Mixed Dataset
        self.small_mixed_yaml_filename = "prompt_dataset__small_mixed.yml"
    
    def cleanup(self):

        """Cleanly shuts down the vLLM engine to free memory for the next model."""
        
        print(f"[vBench] Cleaning up {self.hf_model}...")
        
        # Destroy the vLLM engine and parallel processes
        destroy_model_parallel()
        
        # Delete the LLM object and force garbage collection
        del self.llm
        gc.collect()
        
        # Empty the GPU/ROCm cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prompt (self, prompts: list[str], sampling_params: SamplingParams):
        return self.llm.generate(prompts, sampling_params)
    
    def loadDataset (self, dataset_type: str) -> None:

        '''
        dataset_type: options are "random" or "small-mixed". 
        '''   

        if dataset_type == "random":
            self.dataset_type = "random"
        else:
            self.dataset_type = "small-mixed"
            dataset_path = self.location.joinpath(self.small_mixed_yaml_filename)
            if not dataset_path.exists():
                print(f"[vBench]    Error: Could not load file, {self.small_mixed_yaml_filename} not found.")
            
            with open(dataset_path, 'r') as f:
                # Load the YAML content into a Python dictionary
                config = yaml.safe_load(f)
            
            # 3. Pull the "prompts" list out
            # .get() prevents a crash if 'prompts' is missing from the file
            self.dataset = config.get('prompts', [])
            
            print(f"[vBench]   Successfully loaded {len(self.dataset)} prompts from {dataset_path.resolve()}!")

    def samplePromptVector (self, batch_size: int=1, random_input_length: int|None=None) -> list[str]:
        
        """
        Generates a batch of prompt strings based on the configured dataset type.

        If the dataset type is set to 'random', it produces synthetic prompts using 
        random tokens from the model's vocabulary. Otherwise, it samples from the 
        loaded YAML or JSON dataset.

        Args:
            batch_size (int): The number of prompts to generate in this batch. Defaults to 1.
            random_input_length (int, optional): The exact number of tokens for each 
                generated input prompt. Required if using 'random' dataset.

        Returns:
            list[str]: A list of prompt strings ready for inference.
        """

        output_vector = []

        if self.dataset_type == 'random':

            random_input_length = 256 if not random_input_length else random_input_length
            
            for _ in range(batch_size):
                # Sample random tokens from tokenizer vocabulary and decode to string
                random_ids = [randint(0, self.vocab_size - 1) for _ in range(random_input_length)]
                random_prompt = self.tokenizer.decode(random_ids)
                output_vector.append(random_prompt)
            
        elif self.dataset_type == 'small-mixed':

            output_vector = sample(self.dataset, batch_size)
        
        return output_vector


# ================= Benchmark methods =================
def singleBenchmark (bench: BaseBench,
                     batch_size: int=1,
                     num_tokens: int=2000,
                     dataset_type: str="random",
                     input_length: int = 256,
                     output_length: int = 256,
                     warmup_runs: int = 5,
                     temperature: float=0,
                     top_p: float=1.0,
                     device_type: str="GPU",
                     device: str="n/a",
                     docker_image: str="n/a") -> dict[str, int|float|str]:
    
    # Load the dataset
    bench.loadDataset(dataset_type)
    
    # Fix the sampling parameter
    sampling_params = SamplingParams(
        max_tokens=output_length,
        temperature=temperature,
        top_p=top_p
    )

    # Run warmups
    print('[vBench]   warming up ...')
    for _ in range(warmup_runs):
        prompts = bench.samplePromptVector(batch_size, input_length)
        _ = bench.prompt(prompts, sampling_params=sampling_params)

    # Iterate
    tokens_processed = 0
    tokens_generated = 0
    requests    = 0
    tpots       = []
    ttfts       = []
    itls        = []
    batch_latencies = []
    print('\n\n[vBench]   Benchmark started ...')
    while (tokens_processed < num_tokens):

        # Generate new batch of prompts
        prompts = bench.samplePromptVector(batch_size, input_length)

        # Forward propagation
        start = perf_counter()
        outputs = bench.prompt(prompts, sampling_params)
        stop = perf_counter()

        # Measure total latency for whole batch propagation (real e2e)
        batch_latency = stop - start
        batch_latencies.append(batch_latency)

        # Count total tokens generated and total generation time for the batch
        batch_token_intervals = 0
        batch_decode_duration = 0 # cumulative decoding duration of all requests

        # Iterate over all requests in batch to create a batch mean
        for output in outputs:

            # request_duration = output.metrics.last_token_ts - output.metrics.scheduled_ts
            generated_len_request = len(output.outputs[0].token_ids) 

            # Avoid zero division by skipping faulty request outputs with 0 or 1 tokens generated
            if generated_len_request <= 1: continue

            tokens_generated += generated_len_request

            # Request TTFT
            ttft_request = output.metrics.first_token_ts - output.metrics.scheduled_ts
            ttfts.append(ttft_request)

            # Formula for request TPOT
            decode_duration_request = output.metrics.last_token_ts - output.metrics.first_token_ts
            tpot_request = decode_duration_request * 1e3 / (generated_len_request - 1)
            tpots.append(tpot_request)

            # Denote the number of generated tokens in this request
            batch_token_intervals += generated_len_request - 1
            batch_decode_duration += decode_duration_request

        # Denote latency for the request
        if batch_token_intervals == 0: continue
        itl = batch_decode_duration / batch_token_intervals * 1e3 # convert to ms
        itls.append( itl )    

        # Increment total tokens processed
        if dataset_type == 'random':
            # If dataset is randomly generated, the total number of input tokens is fixed
            tokens_processed += input_length * batch_size
        elif dataset_type == 'small-mixed':
            # Need to count input tokens that were sampled from the small dataset
            for p in prompts:
                tokens_processed += len(bench.tokenizer.encode(p))

        # Increment the number of generated tokens and requests
        requests += batch_size
        
    print('\n\n[vBench]   Benchmark finished.')

    e2e_latency = sum(batch_latencies)

    # Evaluate TTFT statistics
    n = len(ttfts)
    ttfts.sort()
    ttft_mean = sum(ttfts) / n
    ttft_median = ttfts[n // 2]
    ttft_p99 = ttfts[int(0.99 * (n - 1))]

    # Evaluate TPOT statistics
    n = len(tpots)
    tpots.sort()
    tpot_mean = sum(tpots) / n
    tpot_median = tpots[n // 2]
    tpot_p99 = tpots[int(0.99 * (n - 1))]

    # Evaluate ITL statistics
    n = len(itls)
    itls.sort()
    itl_mean = sum(itls) / n
    itl_median = itls[ n // 2 ]
    itl_p99 = itls[int(0.99 * (n - 1))]

    # Evalutate throughput
    throughput = (tokens_processed + tokens_generated) / e2e_latency

    # Collect results
    benchmark_results = {
        "model": bench.hf_model,
        f"{device_type}": device,
        "max_input_length": input_length,
        "max_output_length": output_length,
        "throughput": round(throughput, 2),
        "tpot_mean": round(tpot_mean, 4),
        "tpot_median": round(tpot_median, 4),
        "tpot_p99": round(tpot_p99, 4),
        "ttft_mean": round(ttft_mean, 4),
        "ttft_median": round(ttft_median, 4),
        "ttft_p99": round(ttft_p99, 4),
        "itl_mean": round(itl_mean, 4),
        "itl_median": round(itl_median, 4),
        "itl_p99": round(itl_p99, 4),
        "concurrency": batch_size,
        "precision": str(bench.dtype),
        "docker_image": docker_image,
        "total_benchmark_time": round(e2e_latency, 2),
        "total_requests": int(requests),
        "total_tokens_processed": int(tokens_processed),
        "total_tokens_generated": int(tokens_generated),
    }

    return benchmark_results

def integratedBenchmark (device_type: str="GPU",
                         device: str="n/a",
                         hf_models: list[str]=["inceptionai/jais-13b-chat"],
                         hf_token: str|None=None,
                         batch_sizes: list[int]=[1,4,8,16,32,64,128],
                         num_tokens: int=2000,
                         dataset_type: str="random",
                         input_lengths: list[int] = [256, 512],
                         output_lengths: list[int] = [256, 512],
                         warmup_runs: int = 5,
                         temperature: float=0,
                         top_p: float=1.0,
                         csv_output_path: str|Path|None=None,
                         docker_image: str="n/a") -> dict[str, int|float|str]:
    
    output_rows: list[dict] = []
    test_case_run = 0
    total_test_cases = len(hf_models) * len(input_lengths) * len(output_lengths) * len(batch_sizes)

    for model in hf_models:

        bench = BaseBench(model, hf_token=hf_token, device_type=device_type)

        # Load the dataset
        bench.loadDataset(dataset_type)

        for input_len in input_lengths:

            for output_len in output_lengths:

                for batch_size in batch_sizes:
                    
                    # Print progress
                    progress = round(test_case_run / total_test_cases * 100, 1)
                    print(f"[vBench]   Running Benchmark... ({progress}% completed)\
                                       Model: {model}  Batch_Size: {batch_size}  Input_len: {input_len}  Ouput_len: {output_len}")

                    results = singleBenchmark(bench, 
                                              batch_size, 
                                              num_tokens, 
                                              dataset_type, 
                                              input_len, 
                                              output_len, 
                                              warmup_runs, 
                                              temperature, 
                                              top_p,
                                              device_type,
                                              device,
                                              docker_image)

                    output_rows.append(results)
                    test_case_run += 1

        # Clean up to avoid memory exhaustion
        bench.cleanup()
        del bench

    # Cast rows to csv
    delimiter = ','
    headers = list(output_rows[0].keys())
    csv_string = delimiter.join(headers) + '\n'
    for row in output_rows:
        csv_string += delimiter.join(map(str, row.values())) + '\n'

    # Save
    if not csv_output_path:
        csv_output_path = Path(__file__).resolve().parent.joinpath('new_benchmark_output.csv')
    with open(csv_output_path, "w+") as f:
        f.write(csv_string)

def show (results: dict[str, int|float|str]) -> None:

    # Console Output
    print('-------- Benchmark Details --------')
    print('GPU Device Name:                  ', torch.cuda.get_device_name(0))
    print('HF Model Path:                    ', results['model'])
    print('Warmup Iterations:                ', results['warmups'])
    print('Max. Input Tokens:                ', results['max_input_length'])
    print('Max. Output Tokens:               ', results['max_output_length'])
    print('Batch Size - Concurrency:         ', results['concurrency'])
    print('======== Benchmark Results ========')
    print("Latency Total (seconds):          ", round(results['total_benchmark_time'], 3))
    print("Total Requests:                   ", results['total_requests'])
    print("Mean TTFT (seconds):              ", round(results['ttft_mean'], 3))
    print("Median TTFT (seconds):            ", round(results['ttft_median'], 3))
    print("P99 TTFT (seconds):               ", round(results['ttft_p99'], 3))
    print("Mean TPOT (seconds):              ", round(results['tpot_mean'], 3))
    print("Median TPOT (seconds):            ", round(results['tpot_median'], 3))
    print("P99 TPOT (seconds):               ", round(results['tpot_p99'], 3))
    print("Generated Tokens Total:           ", results['total_tokens_generated'])
    print("Mean Token Throughput (tokens/s): ", round(results['throughput'], 3))