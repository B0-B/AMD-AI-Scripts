# AMD Stable-Diffusion-XL-Base-1.0 Overview

## Overview
A setup guide for inference and training of SDXL 1.0, tested on AMD CDNA architecture.

## Docker Orchestration

To spin up a container we recommend the following [dockerfile](dockerfile) which is based on a rocm/pytorch container base image which was optimized for training. On top layers were added which are necessary for diffusion testing. 
The container contains the following packages for diffuser onboarding

- rocm/pytorch
- mlperf:latest
- diffusers
- transformers 
- accelerate 

and for data mangement and image processing

- pandas 
- torchvision 
- webdataset 
- img2dataset
  
To start the container we recommend to create two directories on the host system (if not exist already):

1. A local huggingface directory via [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) to store large files like models, datasets etc. this will reduce loading and thus waiting times and keeps all files central without redundancy. It is assumed that the directory is located under ``/storage/huggingface`` - change to your needs
2. A local project directory e.g. in your user home directory - here we use `~/sdxl-testing`.

copy the dockerfile into the project directory

```bash
cd ~/sdxl-testing && wget https://raw.githubusercontent.com/B0-B/AMD-AI-Scripts/refs/heads/main/AMD_SDXL-XL-Base-1.0/dockerfile
```

next build the container

```bash
docker build .
```

Both directories will be mounted during start into the container and be located at the root level. The first mount is to access large data and the second to exchange scripts for testing. Also, the mounted huggingface directory will be set as default in the container's environment variable i. e. `HF_HOME`, this will allow to synchronize model storage etc. and reduce redundant downloads in each container which is spun. All points are packaged in the following start command below

```bash
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  -v ~/sdxl-testing:/sdxl-testing \
  -v /storage/huggingface:/huggingface \
  -e HF_HOME=/huggingface \
  sdxl-mlperf bash
```


## Inference
For inference copy the script [sdxl_inference_test.py](sdxl_inference_test.py) into your host project directory `~/sdxl-testing`. Next, spawn the container and within access the project directory

```bash
root@container> cd /sdxl-testing
```

and execute the script 

```bash
python sdxl_inference_test.py
```

The result will be located in the project's root directory under ``~/sdxl-testing/sdxl_output.png``.

<img src=image.png>