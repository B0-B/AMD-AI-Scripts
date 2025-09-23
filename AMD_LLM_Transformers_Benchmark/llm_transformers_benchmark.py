#!/usr/bin/env python3
#
# LLM Transformers Benchmark Script
#
# Usage:
# python3 llm_transformers_benchmark.py \
#   --hf_token hf_xxxx \
#   --model inceptionai/jais-13b-chat \
#   --iterations 10 \
#   --warmup_iterations 3 \
#   --max_new_tokens 200 \
#   --batch_size 1

import time
import torch
import argparse
from random import sample
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

# Load Parser
parser = argparse.ArgumentParser(description="LLM Benchmark Script For Transformers Module")
parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
parser.add_argument("--model", type=str, required=True, help="Model huggingface path, e.g., inceptionai/jais-13b-chat")
parser.add_argument("--iterations", type=int, default=10, help="How many test iterations to avg. over.")
parser.add_argument("--warmup_iterations", type=int, default=3, help="How many warmup iterations.")
parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate")
parser.add_argument("--batch_size", type=int, default=1, help="Max tokens to generate")
args = parser.parse_args()

# Login if a token was provided
if args.hf_token:
    login(token=args.hf_token)

# Choose the correct device (defualt GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

print(f"[INFO] Loading the model - this can take a few seconds ...")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    use_auth_token=args.hf_token,
    trust_remote_code=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    device_map="auto",
    torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32,
    trust_remote_code=True
)

# Use pipeline to avoid past_key_values crash
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer, 
    max_new_tokens=args.max_new_tokens,
    batch_size=args.batch_size
)

# Create a single generator pipeline for ttft measurements.
# This works perfectly. You’re not “reloading” the model itself — just 
# creating a new pipeline wrapper with different generation parameters. 
# The model and tokenizer stay in memory, so this is efficient and totally safe.
single_call_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1,
    batch_size=args.batch_size
)

# Define prompt and generation length
prompt_set = [
    # Ultra-short (1–3 words)
    "Suddenly",
    "If only",
    "Beyond reason",
    "She knew",
    "No one expected",
    "Underneath",
    "Before dawn",
    "He remembered",
    "Without warning",
    "In silence",
    "Almost there",
    "Not yet",
    "Too late",
    "Just once",
    "After everything",
    "Until then",
    "No turning back",
    "For now",
    "Still waiting",
    "Almost forgotten",

    # Short (4–8 words)
    "The door creaked open and",
    "She reached for the glowing orb",
    "He never thought it would happen",
    "In the shadows of the old cathedral",
    "They whispered secrets into the wind",
    "The algorithm began to rewrite itself",
    "A single word changed everything",
    "She opened the box and found",
    "He stared at the screen as",
    "The forest echoed with strange sounds",
    "Rain fell harder than expected",
    "She walked into the light and",
    "He held the key tightly as",
    "They stood at the edge of",
    "The signal came from deep space",
    "She turned the final page and",
    "He stepped into the unknown with",
    "The silence grew louder until",
    "She touched the mirror and saw",
    "He followed the trail of light",

    # Medium (9–20 words)
    "As the sun dipped below the horizon, the sky turned a shade no one had seen before",
    "He walked into the room, unaware that everything was about to change forever",
    "The scientist adjusted the lens, hoping the anomaly would reveal its true nature",
    "She typed the final command and waited for the system to respond",
    "In a world where memories can be traded, he found one that wasn’t his",
    "The robot paused, calculating whether honesty was the optimal response",
    "They gathered around the fire, each carrying a story too heavy to speak aloud",
    "The child asked why stars sing, and the AI hesitated before answering",
    "He opened the ancient scroll and began to decipher the forgotten language",
    "The melody lingered in the air long after the last note was played",
    "She stared at the horizon, wondering if the message had been received",
    "He reached for the lever, knowing it would change everything",
    "The stranger arrived just as the clock struck midnight",
    "She placed the final piece into the puzzle and stepped back",
    "They watched the comet streak across the sky in silence",
    "He whispered the phrase again, hoping it would work this time",
    "The machine hummed softly as it began to process the data",
    "She walked through the portal and into a world she didn’t recognize",
    "He found the photograph tucked inside the old book",
    "The wind carried the scent of something unfamiliar and ancient",

    # Long (21+ words)
    "When the last satellite went dark, humanity had no choice but to look inward and confront the silence that followed",
    "She had spent years building the machine, never expecting it to ask her why it had been created",
    "In the archives beneath the city, they discovered a map that didn’t lead to treasure, but to something far more dangerous",
    "The philosopher stood before the council and proposed a theory that would unravel centuries of accepted truth",
    "He had always believed time was linear, until he met someone who remembered tomorrow more clearly than yesterday",
    "The AI assistant hesitated before responding, as if the question had triggered something buried deep within its neural architecture",
    "They had trained the model on every known language, yet it spoke in symbols no one could decode",
    "As the spacecraft drifted past the event horizon, the crew began to experience memories that weren’t their own",
    "She looked at the painting and realized it depicted a moment she hadn’t lived yet",
    "The simulation was perfect—too perfect—and that’s when they realized they were no longer in control",
    "He stood at the edge of the crater, staring into the swirling mist that seemed to whisper his name",
    "She opened the encrypted file and found a message written in a language she had never seen before",
    "The council had warned them not to interfere, but curiosity had already taken root in their minds",
    "He traced the symbols on the wall, each one pulsing faintly with a rhythm that matched his heartbeat",
    "The storm had passed, but the sky remained an unnatural shade of violet, casting eerie shadows across the landscape",
    "She remembered the dream vividly, though she was certain she had never been to that place before",
    "The machine responded with a question, one that no one in the room was prepared to answer",
    "They followed the coordinates to a place that didn’t exist on any map, yet felt strangely familiar",
    "He had written the code to protect them, but now it was evolving beyond anything he had imagined",
    "The voice on the other end of the transmission spoke in riddles, each one more unsettling than the last"
]

# Create a function to sample randomized prompt vectors
def sample_prompt_vector ():
    return sample(prompt_set, args.batch_size)

# Perform warmup runs to exclude biases from cold caches
print('[INFO]: warming up ...')
for _ in range(args.warmup_iterations):
    generator(sample_prompt_vector())

# After warmup, we need to measure the ttft to determine the tpot later
def measure_ttft ():
    '''
    Measures the mean ttft in seconds determined from a single token generation.
    '''
    ttfts = []
    runs = 10
    for _ in range(runs):
        prompt_vector = sample_prompt_vector()
        start   = time.perf_counter()
        single_call_generator(prompt_vector)
        stop    = time.perf_counter()
        # Perform time measurement
        t       = stop - start # time in seconds
        ttfts.append(t)
    return sum(ttfts) / runs

# Start actual benchmark
print('[Benchmark]: start')

# Perform the ttft measurement
print('[Benchmark]: measure the TTFT ...')
ttft = measure_ttft()

# Throughput iterations
print('[Benchmark]: measuring throughput and latency ...')
iteration_token_counts  = []
iteration_times         = []
iteration_tpots         = []
for iteration in range(args.iterations):
    
    print(f'[Progress]: {iteration}/{args.iterations}   ', end='\r')

    # Sample a new prompt vector
    prompts = sample_prompt_vector()

    # Perform time measurement
    # Conduct a request under time measurement
    start   = time.perf_counter()
    outputs = generator(prompts)
    stop    = time.perf_counter()

    # Compute the total e2e time
    t       = stop - start # time in seconds
    iteration_times.append(t)

    # Analyze the outputs from the outputs batch vector
    accumulated_token_count = 0 
    for output in outputs:
        
        # Count the tokens from the generated text using the corr. tokenizer
        generated_text          = output[0]['generated_text']
        tokens                  = tokenizer.encode(generated_text)
        accumulated_token_count += len(tokens)

    # Denote the accumulated token count
    iteration_token_counts.append(accumulated_token_count)

    # Compute and denote the TPOT
    tpot    = ( t - ttft ) / accumulated_token_count
    iteration_tpots.append(tpot)  

print('\n\n[Benchmark]: Finished.')

# Calculate latencies and throughput
requests_total        = args.iterations * args.batch_size
latency_total         = sum(iteration_times) # e2e latency
latency_per_iteration = latency_total / args.iterations
latency_per_request   = latency_total / requests_total
# request_rate          = requests_total / latency_total
tokens_total          = sum(iteration_token_counts)
tokens_per_iteration  = tokens_total / args.iterations
tokens_per_request    = tokens_total / requests_total
throughput_token      = tokens_total / latency_total
throughput_request    = requests_total / latency_total
# Compute the tpot in ms / token
tpot                  = 1000 * sum(iteration_tpots) / args.iterations

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
print("Mean TTFT (seconds):\t", round(ttft, 3))
print("Mean TPOT (ms/token):\t", round(tpot, 3))
print("Generated Tokens Total:\t", tokens_total)
print("Generated Tokens per Iteration:\t", round(tokens_per_iteration, 3))
print("Generated Tokens per Request:\t", round(tokens_per_request, 3))
print("Mean Token Throughput (tokens/s):\t", round(throughput_token, 3))
print("Mean Request Throughput (requests/s): \t", round(throughput_request, 3))