#!/usr/bin/env python3
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
import os

def main():

    # Check for ROCm support
    if not torch.cuda.is_available():
        raise RuntimeError("No ROCm-capable GPU detected by PyTorch!")

    # Device should map to HIP backend (ROCm)
    device = torch.device("cuda")

    # Load SDXL 1.0 pipeline (fp16 recommended for speed)
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    # Move pipeline to GPU
    pipe = pipe.to(device)

    # Optional: enable memory optimizations
    pipe.enable_attention_slicing()

    # Prompt
    prompt = "A steampunk spaceship between 2 planets and a lot of stars"

    # Generate image
    image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

    # Save output
    out_path = "sdxl_output.png"
    image.save(out_path)
    print(f"Saved generated image to {out_path}")

if __name__ == "__main__":
    main()