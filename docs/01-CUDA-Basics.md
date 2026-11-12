# CUDA Basics: Getting Started with GPU Computing

## What is CUDA?

CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) created by NVIDIA. It allows developers to write programs that execute on NVIDIA GPUs, enabling massive parallelism for computationally intensive tasks.

## Why Use CUDA?

- **Massive Parallelism**: GPUs have thousands of cores compared to CPUs' dozens
- **High Throughput**: Process millions of operations simultaneously
- **Energy Efficiency**: Perform computations faster with lower power consumption
- **Real-world Applications**: Machine Learning, Scientific Computing, Financial Analysis, Graphics Processing

## CUDA Programming Model

### 1. CPU (Host) and GPU (Device)

- **Host**: Traditional multi-core CPU that runs your main program
- **Device**: NVIDIA GPU that executes parallel kernels

Communication happens between host and device through PCIe or NVLink.

```
┌──────────────────┐
│   CPU (Host)     │
│  - Control Logic │
│  - Memory (RAM)  │
└────────┬─────────┘
         │
      PCIe/NVLink
         │
┌────────▼─────────┐
│   GPU (Device)   │
│  - Compute Cores │
│  - GPU Memory    │
└──────────────────┘
```

### 2. Heterogeneous Computing

CUDA programs follow a heterogeneous model:
1. **Host code** (CPU): Sequential execution, controls flow
2. **Device code** (GPU): Parallel kernel execution
3. **Data Transfer**: Between host and device memory

### 3. Basic Execution Flow

```
1. Allocate memory on GPU
2. Copy data from CPU to GPU
3. Launch kernel on GPU
4. Wait for completion
5. Copy results back to CPU
6. Free GPU memory
```

## CUDA Memory Hierarchy

```
┌─────────────────────────────────────────┐
│      CPU Main Memory (Host)             │
│      - Large (GB)                       │
│      - Slower                           │
└────────────────┬────────────────────────┘
                 │ PCIe Transfer
┌────────────────▼────────────────────────┐
│      GPU Global Memory                  │
│      - Large (48GB for H100)            │
│      - Slower (but parallel access)     │
└─────────────────────────────────────────┘
         ▲              ▲              ▲
         │              │              │
    ┌────┴────┐    ┌────┴────┐    ┌───┴─────┐
    │ Shared  │    │  Local  │    │Constant │
    │ Memory  │    │ Memory  │    │ Memory  │
    │ (48-96KB)   │ (per thread) │ (64KB)  │
    └─────────┘    └─────────┘    └─────────┘
```

## Key Concepts Overview

| Concept | Description |
|---------|-------------|
| **Kernel** | Function executed in parallel on GPU |
| **Thread** | Individual execution unit on GPU |
| **Block** | Collection of threads that execute together |
| **Grid** | Collection of blocks that form one kernel launch |
| **Warp** | 32 threads executed in lock-step on SM |
| **SM (Streaming Multiprocessor)** | Processing core with multiple warps |

## Simple CUDA Program Structure

```cuda
#include <stdio.h>
#include <cuda.h>

// Kernel: executes on GPU
__global__ void kernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx * 2;
}

int main() {
    int n = 1024;
    int *d_data;
    
    // Allocate GPU memory
    cudaMalloc((void**)&d_data, n * sizeof(int));
    
    // Launch kernel with 1 block of 256 threads
    kernel<<<1, 256>>>(d_data);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Free GPU memory
    cudaFree(d_data);
    
    return 0;
}
```

## CUDA Data Types and Qualifiers

### Function Qualifiers

| Qualifier | Location | Callable From | Used For |
|-----------|----------|---------------|----------|
| `__global__` | Device | Host | Kernel functions |
| `__device__` | Device | Device | Helper functions on GPU |
| `__host__` | Host | Host | CPU functions (default) |

### Variable Qualifiers

| Qualifier | Type | Scope | Speed |
|-----------|------|-------|-------|
| `__shared__` | Shared Memory | Block | Very Fast |
| `__constant__` | Constant Memory | Global | Fast |
| `__device__` | Global Memory | Global | Moderate |

## Compute Capability

Different NVIDIA GPUs have different compute capabilities:

| GPU | Compute Capability | Cores | Memory |
|-----|-------------------|-------|--------|
| **H100** | 9.0 | 16,896 | 80GB HBM3 |
| **A100** | 8.0 | 6,912 | 40GB HBM2 |
| **RTX 4090** | 8.9 | 16,384 | 24GB GDDR6X |
| **RTX 4080** | 8.9 | 9,728 | 16GB GDDR6X |

Higher compute capability = more features and better performance.

## Next Steps

- [Learn about Kernels, Blocks, and Threads](02-Kernels-Blocks-Threads.md)
- [Understand Indexing and Memory Access](03-Indexing-Memory.md)
- [Set up SLES15 SP7](04-SLES15-Setup.md)
