# `hipBlasLt` Python API — Full Documentation

> A thin, ergonomic wrapper around the `hipblaslt-bench` client for building and running GEMM (matrix multiply) benchmarks on AMD ROCm using **hipBLASLt**.

---


## Overview

AMD's [hipBLASlt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html) is a toolbox within the ROCm software suite for steering and performing linear algebra calculations, mainly centered around matrix-matrix calculus on AMD GPUs. According to the AMD authors, it has a flexible API that extends functionalities beyond a traditional BLAS library, such as adding flexibility to matrix data layouts, input types, compute types, and algorithmic implementations and heuristics. 

In order to operate some of the most popular workloads today it is necessary to understand what they have in common. Most scientific fields, HPC, rendering, and especially AI workloads rely on fundamental tensor calculus, mathematical operations which can be accelerated through parallelization. The most abstract formula which can reproduce all commonly known algorithms can be expressed as

$$D=Activation(α⋅op(A)⋅op(B)+β⋅op(C)+bias) \,\,[1]$$

In the following we will explore every parameter in detail. The equation can be disambled into the terms

- Activation, which a free to choose non-linear activation function, applied to each dimension of the argument vector
- $op()$ is a tensor operation for transposition
- $\alpha \cdot op(A)\cdot op(B)$ which is a pure GEMM calculation
- $\beta\cdot op(C)$ which is a linear feedback term by incoorporating information from another or previous state, can also be understood as momentum
- $bias$ is a constant bias tensor, something like a systematic deviation or a neuron bias. The greater the bias the smaller the exploration and ability to learn or adapt.

and the variables are 

- $\alpha, \beta$ are scalar factors which operate on their corr. matrix or tensor
- $A, B$ are tensors whose size is known as problem sizes: $m$ <int>: rows of $C$ (and $A$), $n$ <int>: cols of $C$ (and $B$), $k$ <int>: cols of $A$ / rows of $B$.
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


## Usage

### Examples

#### 1) Minimal FP32 GEMM

```python
hipblt = hipBlasLt("AMD Instinct MI300X")
hipblt.specifyMatrix("A", 4096, 2048, precision=32)   # A(m,k)
hipblt.specifyMatrix("B", 2048, 8192, precision=32)   # B(k,n)
hipblt.setComputePrecision(32)                        # f32_r
hipblt.specifyScalars(alpha=1.0, beta=0.0)
hipblt.setActivation("none")
hipblt.run(batchSize=1, validate=True)
hipblt.showResults("hipblaslt-Gflops", "hipblaslt-GB/s", "us")
```

#### 2) Mixed Precision (A=f32, B=f16) with FP32 compute

```python
hipblt = hipBlasLt("AMD Radeon RDNA3")
hipblt.specifyMatrix("A", 4096, 2048, precision=32)       # f32_r
hipblt.specifyMatrix("B", 2048, 8192, precision=16)       # f16_r
hipblt.specifyScalars(1.0, 1.0)
hipblt.setComputePrecision(32)                            # compute_type=f32_r (recommended for mixed)
hipblt.setActivation("none")
hipblt.run(batchSize=1, validate=False)
hipblt.showResults("hipblaslt-Gflops", "hipblaslt-GB/s", "us")
```

#### 3) Bias and Activation

```python
hipblt = hipBlasLt("AMD Instinct MI300X")
hipblt.specifyMatrix("A", 2048, 2048, precision=32)
hipblt.specifyMatrix("B", 2048, 2048, precision=16)
hipblt.addBias(precision=32, source="d")                 # bias_type=f32_r
hipblt.setActivation("relu")
hipblt.setComputePrecision(32)
hipblt.specifyScalars(alpha=1.0, beta=1.0)
hipblt.run(validate=True)
hipblt.showResults("hipblaslt-Gflops", "us")
```

#### 4) BFloat16 (BF16) and Batch GEMM

```python
hipblt = hipBlasLt("AMD GPU")
hipblt.specifyMatrix("A", 8192, 4096, precision=16, bfloat=True)  # bf16_r
hipblt.specifyMatrix("B", 4096, 8192, precision=16, bfloat=True)  # bf16_r
hipblt.setComputePrecision(32)                                    # accumulate in f32
hipblt.specifyScalars(1.0, 0.0)
hipblt.setActivation("none")
hipblt.run(batchSize=4, validate=False)                           # batched GEMM
hipblt.showResults("hipblaslt-Gflops", "hipblaslt-GB/s", "us")
```

#### 5) INT8 GEMM (experimental)

```python
hipblt = hipBlasLt("AMD Radeon 9700")
hipblt.specifyMatrix("A", 2048, 2048, precision=8)   # i8_r
hipblt.specifyMatrix("B", 2048, 2048, precision=8)   # i8_r
hipblt.specifyScalars(1.0, 0.0)
# Many INT8 configs expect compute_type=32i; set explicitly if your bench requires it.
hipblt.setComputePrecision(8)  # convertPrecision -> "i8_r"; adjust if you need "32i"
hipblt.setActivation("none")
hipblt.run(validate=False)
hipblt.showResults("hipblaslt-Gflops", "us")
```

### Result Parsing & Display

### ``showResults():``

- Uses a simplified parser tailored to the current bench output style.
- Prints a compact table with requested metrics and a run-info block.
- If parsing fails, consider printing self.results directly and adjusting the parsing logic.


The output will look like

```
--------Benchmark results--------
GPU                       : AMD ...
Matrix Dimensions (m,n,k) : 128, 128, 128      
Precisions (A,B,C,D,bias) : f32_r, f32_r, f32_r, f32_r,
Batch Size                : 1
Alpha                     : 1
Beta                      : 1
Bias Vector               : No
Transpose (A, B)          : [False, False]     

hipblaslt-Gflops    hipblaslt-GB/s      us                  CPU-Gflops          CPU-us        
270.6               15.751              15.5                0.125507            33419
```

### Class Reference

`hipBlasLt` is a convenience class that:

- Builds **reproducible** `hipblaslt-bench` command lines for GEMM runs.
- Handles **precision strings** (e.g., `f16_r`, `bf16_r`, `f32_r`, `i8_r`) for A/B/C/D and compute type.
- Manages **transpose flags** and **leading dimensions** (as currently implemented: defaults to non-transposed leading dimensions).
- Supports **bias** epilogue and **activation** selection.
- Runs the benchmark and **parses output** to display selected metrics.
  
### ``hipBlasLt.__init__(gpuName="")``


Create the wrapper.

#### Parameters

gpuName (str, optional): A friendly GPU name for display (e.g., "AMD Instinct MI300X").

#### Attributes (selected)

- matrixDimensions (dict): Shapes for A/B, stored as (rows, cols) pre‑transpose.
- matrixPrecision (dict): Precision strings for A/B/C/D (defaults to FP32).
- transpose (dict): Boolean transpose flags for A/B.
- alpha, beta (float): Scalar multipliers.
- activation (str): Epilogue activation (default: "none").
- computePrecision (str): Compute type string (e.g., "f32_r").
- biasPrecision, biasSource (str): Epilogue bias configuration.
- batchSize (int): Last run’s batch size.
- results (str): Raw captured output from hipblaslt-bench.

---

### ``specifyMatrix(matrixName, rows, cols, precision, transpose=False, bfloat=False)``

Define matrix shape and precision. Shapes are pre‑transpose.

#### Parameters

- matrixName (str): 'A' or 'B' (case-insensitive).
- rows (int): Number of rows (must be positive).
- cols (int): Number of columns (must be positive).
- precision (int): Bit width, typically 16 or 32 (and 8 for INT8).
- transpose (bool, optional): Whether to mark the matrix as transposed (False → N, True → T).
- bfloat (bool, optional): When precision == 16, produce BF16 (bf16_r) if True.

#### Behavior

- Converts precision to a hipBLASLt string via convertPrecision().
- Stores (rows, cols) and transpose flag.
- Validates A.k == B.k when both matrices are known.

#### Raises

- KeyError: If matrixName is not one of A, B.
- ValueError: For unsupported precision, non‑positive dims, or mismatched shapes.

---

### ``addBias(precision, bfloat=False, source='d')``

Enable bias epilogue for the GEMM.

#### Parameters

- precision (int): Bias precision (e.g., 32, or 16 for FP16/BF16).
- bfloat (bool, optional): If precision == 16, set BF16 (bf16_r) when True.
- source (str, optional): Epilogue source location; 'd' is common.

#### Behavior

- Sets --bias_vector --bias_type <...> --bias_source <...> in the command.

---

### ``setActivation(activationFunction)``

Select epilogue activation.

#### Parameters

- activationFunction (str): One of none, gelu, relu, swish, clamp.

#### Raises

- ValueError: If the activation string isn’t supported.

---

### ``specifyScalars(alpha, beta)``

Set GEMM scalars.

#### Parameters

- alpha (float): Multiplier for op(A) * op(B).
- beta (float): Multiplier for op(C).

---

### ``setComputePrecision(precision)``

Set compute type explicitly (accumulation precision).

#### Parameters

- precision (int): Commonly 32 → f32_r.
- For INT8 paths, you may need compute_type=32i (not automatically set by this method).

---

### ``convertPrecision(precision, bfloat=False)``

Convert integer bits to a hipBLASLt precision string.

#### Parameters

- precision (int): Bit width (e.g., 8, 16, 32).
- bfloat (bool, optional): Only applies for precision == 16 (produces bf16_r).

#### Returns

(str): One of i8_r, f16_r, bf16_r, f32_r.

#### Examples

```python
convertPrecision(32)         # "f32_r"
convertPrecision(16)         # "f16_r"
convertPrecision(16, True)   # "bf16_r"
convertPrecision(8)          # "i8_r"
```

---

### ``run(batchSize=1, validate=False, warmupIterations=0)``

Build and execute hipblaslt-bench with stored parameters.

#### Parameters

- batchSize (int, optional): Total GEMMs to compute in parallel (--batch_count).
- validate (bool, optional): If True, adds -v for validation.
- warmupIterations (int, optional): Prepends warmup iterations (currently passed as -j in the command string in your implementation).

#### Behavior

Constructs a command with:

- -m -n -k (from A(m,k) and B(k,n))
- --transA/--transB (N/T)
- Leading dimensions (current default): --lda m --ldb k --ldc m --ldd m
- --alpha --beta
- --a_type --b_type --c_type --d_type
- --compute_type (as previously set)
- --activation_type (as previously set)
- --bias_vector --bias_type --bias_source (if bias enabled)
- --batch_count when batchSize > 1
- -v when validate=True
- -j <warmupIterations> when provided
- Executes the command via subprocess.run(shell=True).
- Stores raw output in self.results (with a simple post-trim in the current code).

#### Errors

Exceptions are caught and printed; on error, self.results may contain stderr.

---

### ``showResults(*metrics)``

Parse hipblaslt-bench output and print selected metrics.

#### Parameters

- metrics (str, variadic): Metric names to display. Examples include: batchcount, hipblaslt-Gflops, hipblaslt-GB/s, us, CPU-Gflops, CPU-us.

#### Behavior

The current implementation:

- Collapses spaces in self.results.
- Splits into header and values (assumes a two-line tail).
- Prints a compact table aligned to a fixed cell width.
- Prints a run info block (GPU, dims, precisions, transpose flags, etc.).

**Note:** If your bench output format differs, you may need to adjust the parser.

---

### ``getResultData(self, *metrics) -> dict[str, str | float | int]``

Collects selected benchmark metrics from the stored self.results string (produced by a hipblaslt-bench run) and returns the raw values (with floats and integers evaluated) in a dictionary.

#### Purpose
getResultData parses the tail of the hipblaslt-bench output to extract a subset of metrics you care about (e.g., hipblaslt-Gflops, hipblaslt-GB/s, us) and returns a mapping from metric name to value for easy programmatic consumption.

#### Parameters

- *metrics: A variable-length list of metric names (strings) to include in the result.

#### Returns

- A dictionary where keys are metric names and values are the parsed values (as strings; you can cast to numbers later).
