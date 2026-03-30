#!/usr/bin/env python3
import os
import sys
# Try importing yaml, otherwise install it on-the-fly
try:
    import yaml
except ImportError:
    # Use sys.executable to target the current python environment
    os.system(f"{sys.executable} -m pip install pyyaml")
    import yaml
from time import perf_counter
import argparse
import torch
from pathlib import Path
from random import sample, randint
from huggingface_hub import login
from vllm import LLM, SamplingParams


def stdDev (values: list[float], mean: float|None=None) -> float:
    n = len(values)
    if not mean:
        mean = sum(values) / n
    return (sum([(values[i] - mean)**2 for i in range(n)]) / (n - 1)) ** .5


# =============== Base Bench Object =================
class BaseBench:

    def __init__(self, hf_model: str, hf_token: str|None=None, device_type: str="GPU"):
        
        self.location = Path(__file__).resolve().parent

        self.hf_model = hf_model

        if hf_token:
            login(token=hf_token)

        self.llm = LLM(
            model=hf_model,
            tokenizer=hf_model,
            trust_remote_code=True,
            tokenizer_mode="auto",
            download_dir=None,
            hf_token=hf_token
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.dtype = self.llm.llm_engine.model_config.dtype

        # Random dataset 
        self.dataset_type: str|None = None
        self.dataset: list[str]|None = None

        # Small Mixed Dataset
        self.small_mixed_yaml_filename = "prompt_dataset__small_mixed.yml"
    
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

    def ttft (self, sampling_params: SamplingParams, batch_size: int=1, random_input_length: int|None=None, probings: int=10) -> list[float, float, float]:
        
        '''
        Measures the mean time-to-first-token, median and standard deviation (in seconds) estimated from a single token generation.
        The TTFT will depend on the input length, the batch size and the general sampling parameter.

        Returns a tuple with mean and deviation
        '''
        
        print(f"[vBench]   Measure TTFT ...")
        ttfts = []
        for _ in range( probings ):
            prompt_vector = self.samplePromptVector(batch_size, random_input_length)
            start   = perf_counter()
            self.prompt(prompt_vector, sampling_params)
            stop    = perf_counter()
            # Perform time measurement
            t       = stop - start # time in seconds
            ttfts.append(t)
        
        ttfts.sort()
        ttft_mean = sum(ttfts) / probings
        ttft_median = ttfts[int(probings/2) if probings % 2 == 0 else int((probings-1)/2+1)]
        ttft_sigma = stdDev(ttfts, ttft_mean) # Use sample variance formula

        return [ttft_mean, ttft_median, ttft_sigma]


# ================= Single benchmark method =================
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

    # Before iteration probe the TTFT statistics
    ttft_mean, ttft_median, ttft_dev = bench.ttft(sampling_params, batch_size, input_length, probings=10)
    ttft_p99 = 2.326 * ttft_dev # In a standard normal distribution the 99th percentile is associated with approximately 2.326 sigma

    # Iterate
    tokens_processed = 0
    tokens_generated = 0
    requests = 0
    latencies = []
    tpots = []
    print('\n\n[vBench]   Benchmark started ...')
    while (tokens_processed < num_tokens):

        # Generate new batch of prompts
        prompts = bench.samplePromptVector(batch_size, input_length)

        # Query
        start = perf_counter()
        outputs = bench.prompt(prompts, sampling_params)
        stop = perf_counter()

        # Denote latency
        t = stop - start
        latencies.append(t)

        # Count output tokens
        accumulated_output_token_count = 0
        for output in outputs:
            generated_text = output.outputs[0].text
            full_text = output.prompt + generated_text
            accumulated_output_token_count += len(bench.tokenizer.encode(full_text))

        # Compute the TPOT and denote it
        tpot = (t - ttft_mean) / accumulated_output_token_count
        tpots.append(tpot)

        # Increment total tokens processed
        if dataset_type == 'random':
            # If dataset is randomly generated, the total number of input tokens is fixed
            tokens_processed += input_length * batch_size
        elif dataset_type == 'small-mixed':
            # Need to count input tokens that were sampled from the small dataset
            for p in prompts:
                tokens_processed += len(bench.tokenizer.encode(p))

        # Increment the number of generated tokens and requests
        tokens_generated += accumulated_output_token_count
        requests += batch_size
        
    print('\n\n[vBench]   Benchmark finished.')

    # Evaluate TPOT statistics
    tpots.sort()
    n = len(tpots)
    tpot_mean = sum(tpots) / n
    tpot_median = tpots[int(n/2) if n % 2 == 0 else int((n-1)/2+1)]
    tpot_sigma = stdDev(tpots, tpot_mean) # Use sample variance formula
    tpot_p99 = 2.326 * tpot_sigma # In a standard normal distribution the 99th percentile is associated with approximately 2.326 sigma

    # Evaluate ITL statistics
    latencies.sort()
    e2e_latency = sum(latencies)
    itl_mean = e2e_latency / n
    itl_median = latencies[int(n/2) if n % 2 == 0 else int((n-1)/2+1)]
    itl_sigma = stdDev(latencies, itl_mean)  # Use sample variance formula
    itl_p99 = 2.326 * itl_sigma # In a standard normal distribution the 99th percentile is associated with approximately 2.326 sigma

    # Collect results
    benchmark_results = {
        "model": bench.model,
        f"{device_type}": device,
        "max_input_length": input_length,
        "max_output_length": output_length,
        "throughput": tokens_generated / e2e_latency,
        "tpot_mean": tpot_mean,
        "tpot_median": tpot_median,
        "tpot_p99": tpot_p99,
        "ttft_mean": ttft_mean,
        "ttft_median": ttft_median,
        "ttft_p99": ttft_p99,
        "itl_mean": itl_mean,
        "itl_median": itl_median,
        "itl_p99": itl_p99,
        "concurrency": batch_size,
        "precision": bench.dtype,
        "docker_image": docker_image,
        "total_benchmark_time": e2e_latency,
        "total_requests": requests,
        "total_tokens_processed": tokens_processed,
        "total_tokens_generated": tokens_generated,
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

    # Cast rows to csv
    delimiter = ','
    csv_string = delimiter.join(list(output_rows[0].keys())) + '\n'
    for row in output_rows:
        csv_string += delimiter.join(list(row.values())) + '\n'

    # Save
    if not csv_output_path:
        csv_output_path = Path(__file__).resolve().parent.joinpath('new_benchmark_output.csv')
    with open(csv_output_path) as f:
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