
# `hipBlasLt` Python API — Full Documentation

> A thin, ergonomic wrapper around the `hipblaslt-bench` client for building and running GEMM (matrix multiply) benchmarks on AMD ROCm using **hipBLASLt**.

---

## Table of Contents

- Overview
- Quick Start
- Installation & Requirements
- Concepts
  - GEMM Anatomy
  - Data Types & Precision Strings
  - Transpose & Leading Dimensions
  - Epilogue: Bias & Activation
- Module Layout
- Class Reference
  - `hipBlasLt.__init__(gpuName="")`
  - `specifyMatrix(matrixName, rows, cols, precision, transpose=False, bfloat=False)`
  - `addBias(precision, bfloat=False, source='d')`
  - `setActivation(activationFunction)`
  - `specifyScalars(alpha, beta)`
  - `setComputePrecision(precision)`
  - `convertPrecision(precision, bfloat=False)`
  - `run(batchSize=1, validate=False, warmupIterations=0)`
  - `showResults(*metrics)`
- Usage Examples
  - 1) Minimal FP32 GEMM
  - 2) Mixed Precision (A=f32, B=f16) with FP32 compute
  - 3) Bias and Activation
  - 4) BFloat16 (BF16) and Batch GEMM
  - 5) INT8 GEMM (experimental)
- Result Parsing & Display
- Troubleshooting
- Best Practices

---

## Overview

`hipBlasLt` is a convenience class that:

- Builds **reproducible** `hipblaslt-bench` command lines for GEMM runs.
- Handles **precision strings** (e.g., `f16_r`, `bf16_r`, `f32_r`, `i8_r`) for A/B/C/D and compute type.
- Manages **transpose flags** and **leading dimensions** (as currently implemented: defaults to non-transposed leading dimensions).
- Supports **bias** epilogue and **activation** selection.
- Runs the benchmark and **parses output** to display selected metrics.

It’s designed to help you quickly iterate on matrix sizes, mixed precision, and epilogue settings—without hand‑crafting command lines and parsing text by hand.

---

## Quick Start

```python
from hipblaslt_api import hipBlasLt  # replace with your module path

hipblt = hipBlasLt(gpuName="AMD Instinct MI300X")
hipblt.specifyMatrix("A", 4096, 2048, precision=32)   # A(m=4096, k=2048)
hipblt.specifyMatrix("B", 2048, 8192, precision=32)   # B(k=2048, n=8192)
hipblt.setComputePrecision(32)                         # compute_type=f32_r
hipblt.specifyScalars(alpha=1.0, beta=0.0)
hipblt.setActivation("none")
hipblt.run(batchSize=1, validate=True)
hipblt.showResults("hipblaslt-Gflops", "hipblaslt-GB/s", "us")
```

## Installation & Requirements

ROCm installed with hipBLASLt and the hipblaslt-bench client available in your PATH.
Python 3.8+.
No external Python packages are required; the wrapper uses the standard library (subprocess, shlex).

Note: The class shells out to hipblaslt-bench. Ensure the binary matches your ROCm/HIP stack and that Tensile kernels for your gfx target are present.


