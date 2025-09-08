#!/usr/bin/env python3
#
# vLLM Benchmark Script for LLM Inference
#
# Usage:
# python3 llm_vllm_benchmark.py \
#   --hf_token hf_xxxx \
#   --model inceptionai/jais-13b-chat \
#   --iterations 10 \
#   --warmup_iterations 3 \
#   --max_new_tokens 200 \
#   --batch_size 1

import time
import argparse
import torch
from random import sample
from huggingface_hub import login
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main ():

    # Load Parser
    parser = argparse.ArgumentParser(description="LLM Benchmark Script For vLLM")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--model", type=str, required=True, help="Model huggingface path, e.g., inceptionai/jais-13b-chat")
    parser.add_argument("--iterations", type=int, default=10, help="How many test iterations to avg. over.")
    parser.add_argument("--warmup_iterations", type=int, default=3, help="How many warmup iterations.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per iteration")
    args = parser.parse_args()

    # Login if a token was provided
    if args.hf_token:
        login(token=args.hf_token)

    print(f"[INFO] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_auth_token=args.hf_token,
        trust_remote_code=True
    )

    print(f"[INFO] Loading model with vLLM ...")
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tokenizer_mode="auto",
        download_dir=None,
        hf_token=args.hf_token
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.9
    )

    # Define prompt and generation length
    prompt_set = [
        "Hello I'm a Language model,",
        "The theory of quantum mechanics and",
        "Have you ever considered",
        "In the beginning, there was only silence.",
        "Artificial intelligence is transforming",
        "The cat jumped over the",
        "Once upon a time in a distant galaxy,",
        "The economic implications of inflation include",
        "If I were invisible for a day, I would",
        "The mitochondrion is known as",
        "She opened the door and saw",
        "The future of transportation involves",
        "Climate change is primarily caused by",
        "The best way to learn a new language is",
        "He looked at the stars and wondered",
        "The recipe calls for two cups of",
        "In a world where robots rule,",
        "The capital of France is",
        "To solve the equation, we must first",
        "The sound of rain on the window made me",
        "The history of the Roman Empire begins with",
        "If dogs could talk, they might say",
        "The book was lying open on the table, revealing",
        "Quantum entanglement suggests that",
        "The best advice I ever received was",
        "She typed the final line of code and",
        "The sun dipped below the horizon as",
        "The concept of time dilation arises from",
        "He walked into the room and immediately noticed",
        "The secret ingredient in the sauce is",
        "The algorithm performs best when",
        "The painting depicted a surreal landscape of",
        "The last message from the spacecraft read",
        "The moral of the story is",
        "The experiment failed because",
        "The stars aligned perfectly when",
        "The robot paused and said",
        "The universe is expanding because",
        "The password was hidden inside",
        "The professor explained that entropy is",
        "The dragon flew over the mountains and",
        "The email contained a mysterious link to",
        "The detective found a clue under",
        "The simulation showed that",
        "The child asked why the sky is",
        "The poet wrote about the beauty of",
        "The astronaut looked out the window and saw",
        "The machine learning model predicted",
        "The philosopher argued that reality is",
        "The haunted house was filled with",
        "The code compiled successfully, but",
        "The violinist played a melody that",
        "The ancient scroll revealed a map to",
        "The AI assistant responded with",
        "The spaceship landed on a planet covered in",
        "The theory of relativity implies that",
        "The hacker bypassed the firewall using",
        "The garden was overgrown with",
        "The knight drew his sword and",
        "The storm clouds gathered above",
        "The teacher asked the students to",
        "The robot's memory contained",
        "The butterfly landed on",
        "The quantum computer solved",
        "The magician pulled a rabbit from",
        "The scientist hypothesized that",
        "The train arrived at the station and",
        "The melody echoed through the",
        "The AI model struggled with",
        "The castle stood atop a hill surrounded by",
        "The message was encoded using",
        "The painter mixed shades of blue and",
        "The spaceship accelerated toward",
        "The detective suspected that",
        "The forest was silent except for",
        "The theory was disproven when",
        "The dog barked at",
        "The sun rose over the desert and",
        "The algorithm failed to detect",
        "The wizard cast a spell that",
        "The stars twinkled above the",
        "The robot's voice sounded like",
        "The book described a world where",
        "The chef prepared a dish with",
        "The student asked a question about",
        "The AI generated a story about",
        "The door creaked open to reveal",
        "The spaceship's engine hummed as",
        "The code snippet included",
        "The philosopher pondered the meaning of",
        "The dragon's breath scorched",
        "The simulation predicted a future where",
        "The violin's strings vibrated with",
        "The hacker used a vulnerability in",
        "The stars formed a pattern resembling",
        "The robot's sensors detected",
        "The theory was supported by evidence from",
        "The castle gates opened to reveal",
        "The AI assistant suggested",
        "The spacecraft entered orbit around",
        "The algorithm optimized for",
        "The poem ended with the line",
        "The scientist discovered a new element called",
        "The knight rode into battle with",
        "The machine translated the text into",
        "The story began with a whisper in the dark"
    ]

    # Create a function to sample randomized prompt vectors
    def sample_prompt_vector ():
        return sample(prompt_set, args.batch_size)

    # Warmup
    for _ in range(args.warmup_iterations):
        prompts = sample_prompt_vector()
        _ = llm.generate(prompts, sampling_params)

    # Start actual benchmark
    print('[Benchmark]: start ...')
    iteration_token_counts  = []
    iteration_times         = []
    for iteration in range(args.iterations):
        print(f'[Progress]: {iteration}/{args.iterations}   ', end='\r')
        prompts = sample_prompt_vector()

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        stop = time.perf_counter()

        t = stop - start
        iteration_times.append(t)

        accumulated_token_count = 0
        for output in outputs:
            generated_text = output.outputs[0].text
            full_text = output.prompt + generated_text
            tokens = tokenizer.encode(full_text)
            accumulated_token_count += len(tokens)

        iteration_token_counts.append(accumulated_token_count)

    print('\n\n[Benchmark]: Finished.')

    # Metrics
    requests_total = args.iterations * args.batch_size
    latency_total = sum(iteration_times)
    latency_per_iteration = latency_total / args.iterations
    latency_per_request = latency_total / requests_total
    tokens_total = sum(iteration_token_counts)
    tokens_per_iteration = tokens_total / args.iterations
    tokens_per_request = tokens_total / requests_total
    throughput_token = tokens_total / latency_total
    throughput_request = requests_total / latency_total

    # Console Output
    print('-------- Benchmark Details --------')
    print('GPU Device Name:\t\t', torch.cuda.get_device_name(0))
    print('HF Model Path:\t\t', args.model)
    print('Total Iterations:\t\t', args.iterations)
    print('Warmup Iterations:\t\t', args.warmup_iterations)
    print('Max. Tokens per Iteration:\t\t', args.max_new_tokens)
    print('Batch Size / Iteration:\t\t', args.batch_size)
    print('======== Benchmark Results ========')
    print("Latency Total (seconds):\t", round(latency_total, 3))
    print("Latency per Batch (seconds):\t", round(latency_per_iteration, 3) )
    print("Latency per Request (s):\t", round(latency_per_request, 3) )
    print("Total Requests or Prompts:\t", requests_total)
    print("Generated Tokens Total:\t", tokens_total)
    print("Generated Tokens per Iteration):\t", round(tokens_per_iteration, 3))
    print("Generated Tokens per Request):\t", round(tokens_per_request, 3))
    print("Mean Token Throughput (tokens/s):\t", round(throughput_token, 3))
    print("Mean Request Throughput (requests/s): \t", round(throughput_request, 3))

if __name__ == '__main__':
    main()