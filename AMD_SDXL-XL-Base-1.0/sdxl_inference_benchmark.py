#!/usr/bin/env python3
#####################################################################################################
#
# Stable Diffusion XL 1.0 Inference Benchmark for AMD Instinct GPUs.
#
# ---------------------------------------------------------------------------------------------------
# Usage:
# Adjust the parameter and execute with
# > python3 sdxl_inference_benchmark.py
# ---------------------------------------------------------------------------------------------------
#
# ======== Benchmark Parameter ========
model_id       = "stabilityai/stable-diffusion-xl-base-1.0"
device         = "cuda" # will work for AMD/ROCm
image_px_size  = 512
batch_size     = 1
num_batches    = 10
num_steps      = 10
precision      = "fp16"
safe_tensors   = True
attn_slicing   = True
guidance_scale = 2
warmup_runs    = 2
#####################################################################################################

import torch
import random, math
from time import perf_counter
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image

def benchmark ():

    # Evaluate Inputs
    if batch_size < 1:
        raise ValueError('batch_size must be >=1.')
    
    # Determine the corr. torch dtype
    precision_dtype = {
        #"fp8" : torch.float8, # not supported with rocm/torch yet
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp64": torch.float64
    }[precision]

    # ---- Prepare Pipeline ----
    # Create the inference pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        variant=precision,
        dtype=precision_dtype,
        use_safe_tensors=safe_tensors
    )
    pipe = pipe.to(device)
    if attn_slicing:
        pipe.enable_attention_slicing()


    # Prompt List
    prompts = [
        "A cyberpunk cityscape at night, neon lights reflecting on wet pavement",
        "Medieval knight riding a dragon through stormy skies",
        "Ein viktorianisches Wohnzimmer mit antiken Möbeln und Kamin",
        "Portrait of a futuristic samurai in glowing armor",
        "Un chat qui joue du piano dans un salon élégant",
        "Surreal dreamscape with floating islands and upside-down waterfalls",
        "A 1920s jazz club, smoky atmosphere, vibrant crowd",
        "Japanese tea ceremony in a tranquil garden, spring blossoms",
        "Una biblioteca antigua iluminada por velas",
        "Hyperrealistic close-up of a honeybee on a sunflower",
        "A steampunk airship flying over a desert canyon",
        "Fantasy forest with bioluminescent plants and glowing creatures",
        "Berlin bei Nacht, Regen, reflektierende Straßen",
        "A minimalist Scandinavian kitchen with natural light",
        "Post-apocalyptic wasteland with ruined skyscrapers and lone survivor",
        "Children playing in a snowy village, vintage style",
        "An astronaut discovering alien ruins on Mars",
        "Fashion editorial: avant-garde outfits in a neon-lit alley",
        "Ein geheimnisvoller Wald mit Nebel und alten Ruinen",
        "A watercolor painting of a Paris street café",
        "Retro-futuristic diner with robots serving milkshakes",
        "A magical library with floating books and glowing runes",
        "Underwater coral reef teeming with colorful fish",
        "Una escena de tango en Buenos Aires, estilo cinematográfico",
        "A cozy cabin in the woods during autumn",
        "Portrait of a woman with butterflies in her hair, surreal style",
        "A bustling Moroccan market at sunset",
        "天空に浮かぶ城と飛行する船",
        "A noir detective office, rain outside, cigarette smoke inside",
        "A Renaissance-style painting of a modern city skyline",
        "A fantasy battle between elves and orcs in a misty valley",
        "Ein modernes Wohnzimmer mit Blick auf die Alpen",
        "A psychedelic landscape with melting clocks and rainbow skies",
        "A high-fashion runway in a futuristic glass dome",
        "A serene lake surrounded by cherry blossoms",
        "Un robot jardinero cuidando flores en un invernadero",
        "A medieval banquet hall filled with nobles and jesters",
        "A sci-fi lab with holographic interfaces and glowing tubes",
        "Ein Porträt eines alten Seemanns mit wettergegerbtem Gesicht",
        "A cozy bookstore with cats lounging on shelves",
        "A surreal collage of clocks, eyes, and staircases",
        "A Viking ship sailing through icy waters under aurora borealis",
        "A romantic gondola ride in Venice at twilight",
        "Un festival de luces en una ciudad futurista",
        "A haunted mansion with flickering candles and creaky floors",
        "A peaceful Zen temple in the mountains",
        "Ein futuristischer Zug, der durch eine Cyberwelt fährt",
        "A child’s imagination: dinosaurs, spaceships, and candy mountains",
        "A baroque ballroom with dancers in elaborate costumes",
        "A digital art piece of a phoenix rising from code"
    ]

    # Batch Iteration
    times = []
    for iter in range(num_batches+warmup_runs):

        # Generate a random prompt vector 
        prompt_batch = random.sample(prompts, batch_size)

        # Batched Inference
        start   = perf_counter()
        pipe( prompt_batch, 
              num_inference_steps=num_steps,
              guidance_scale=guidance_scale, 
              height=image_px_size, 
              width=image_px_size   )
        stop    = perf_counter()

        # Skip if still in warmup
        if iter + 1 < warmup_runs:
            print("Warmup ...")
            continue
        
        print(f'Finished batch {iter - warmup_runs + 1}/{num_batches}')

        latency = stop - start
        times.append(latency)

    # Compute statistics
    latency_mean  = round( sum(times) / num_batches, 3)
    latency_var   = sum([(latency - latency_mean)**2 for latency in times]) / (num_batches - 1) # unbiased sample variance
    latency_sigma = round( math.sqrt(latency_var), 3)

    # Results
    total_time       = round(sum(times), 3)
    images_generated = batch_size * num_batches
    iteration_throughput = round(num_batches / total_time, 3)
    image_throughput = round(images_generated / total_time, 3)

    # ---- Output ----
    print('================ Benchmark Results ================')
    print(f'Batch Size:                       {batch_size}')
    print(f'Total Batches:                    {num_batches}')
    print(f'Precision:                        {precision}')
    print(f'Image Size:                       {image_px_size} x {image_px_size}')
    print(f'Images Generated:                 {images_generated}')
    print(f'Iteration Throughput (it/s):      {iteration_throughput}')
    print(f'Image Throughput (images/s):      {image_throughput}')
    print(f'Total E2E Latency (s):            {total_time}')
    print(f'Avg. Batch Latency in 2σ (s):     {latency_mean} +/- {2*latency_sigma}')

# Run
if __name__ == '__main__':
    benchmark()