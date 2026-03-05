#!/usr/bin/env python3
# A simple demonstration of LLM quantization with Quark tool on AMD CDNA GPUs.
# Copyright (C), 2026 by AMD authors.

# =============== Parameters =================
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
NUM_CALIBRATION_DATA = 512
# ============================================

# Import modules
from pathlib import Path
currentDir = Path(__file__).resolve().parent

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
from torch.utils.data import DataLoader

from quark.torch import ModelQuantizer
from quark.torch.export.api import SafetensorsExporter
from quark.torch.export import ExporterConfig, JsonExporterConfig
from quark.torch.quantization.config.config import QuantizationConfig, Config
from quark.torch.quantization import FP8E4M3PerTensorSpec



# 1.) First, load the pre-trained model and its corresponding tokenizer using the Hugging Face transformers library.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, model_max_length=MAX_SEQ_LEN)
tokenizer.pad_token = tokenizer.eos_token


# 2.) Prepare the calibration DataLoader (static quantization requires calibration data).
dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
text_data = dataset["text"][:NUM_CALIBRATION_DATA]
tokenized_outputs = tokenizer(
text_data, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
calib_dataloader = DataLoader( tokenized_outputs['input_ids'], batch_size=BATCH_SIZE, drop_last=True )

# 3.) Define the quantization configuration. See the comments in the following code 
# snippet for descriptions of each configuration option.
# Define fp8/per-tensor/static spec.
FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
# Define global quantization config, input tensors and weight apply FP8_PER_TENSOR_SPEC.
global_quant_config = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
# Define quantization config for kv-cache layers, output tensors apply FP8_PER_TENSOR_SPEC.
KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC
kv_cache_layer_names_for_llama = ["*k_proj", "*v_proj"]
kv_cache_quant_config = {name : QuantizationConfig(input_tensors=global_quant_config.input_tensors,
                                    weight=global_quant_config.weight,
                                    output_tensors=KV_CACHE_SPEC)
                        for name in kv_cache_layer_names_for_llama}
layer_quant_config = kv_cache_quant_config.copy()
# Create quantization config
EXCLUDE_LAYERS = ["lm_head"]
quant_config = Config(
    global_quant_config=global_quant_config,
    layer_quant_config=layer_quant_config,
    kv_cache_quant_config=kv_cache_quant_config,
    exclude=EXCLUDE_LAYERS)

# 4.) Quantize the model and export
# Apply quantization.
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)
# Freeze quantized model to export.
freezed_model = quantizer.freeze(model)
# Define export config.
LLAMA_KV_CACHE_GROUP = ["*k_proj", "*v_proj"]
export_config = ExporterConfig(json_export_config=JsonExporterConfig())
export_config.json_export_config.kv_cache_group = LLAMA_KV_CACHE_GROUP


EXPORT_SUBDIR = MODEL_ID.split("/")[1] + "-w-fp8-a-fp8-kvcache-fp8-pertensor"
EXPORT_DIR = currentDir.joinpath(EXPORT_SUBDIR)
exporter = SafetensorsExporter(model=model, output_dir=EXPORT_DIR)
with torch.no_grad():
    exporter._export(freezed_model, quant_config=quant_config, tokenizer=tokenizer)
