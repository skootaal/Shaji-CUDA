# Kernels, Blocks, and Threads: The CUDA Execution Model

## Overview

The CUDA execution model is based on a hierarchy of **grids**, **blocks**, and **threads**. Understanding this hierarchy is crucial for writing efficient CUDA programs.

## The Execution Hierarchy

```
┌─────────────────────────────────────────────────┐
│  GRID (One Kernel Launch)                       │
│  ┌─────────────────────────────────────────────┐│
│  │ BLOCK (0,0)       │ BLOCK (0,1)             ││
│  │ ┌──────────────┐  │ ┌──────────────┐       ││
│  │ │ Thread (0,0) │  │ │ Thread (0,0) │       ││
│  │ │ Thread (0,1) │  │ │ Thread (0,1) │       ││
│  │ │ ...          │  │ │ ...          │       ││
│  │ │ Thread (0,n) │  │ │ Thread (0,n) │       ││
│  │ └──────────────┘  │ └──────────────┘       ││
│  │ ┌──────────────┐  │ ┌──────────────┐       ││
│  │ │ BLOCK (1,0)  │  │ │ BLOCK (1,1)  │       ││
│  │ │ ...          │  │ │ ...          │       ││
│  │ └──────────────┘  │ └──────────────┘       ││
│  └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```

## 1. Threads

### What is a Thread?

- **Smallest execution unit** in CUDA
- Each thread executes the same kernel code independently
- Threads have **local variables** and **local memory**
- Threads are identified by a unique index within their block

### Thread Identification

```cuda
// Built-in variables for thread identification
threadIdx.x  // Thread index within block (X dimension)
threadIdx.y  // Thread index within block (Y dimension)
threadIdx.z  // Thread index within block (Z dimension)

// Examples:
// For 1D block: threadIdx.x = 0 to blockDim.x-1
// For 2D block: threadIdx.x and threadIdx.y
// For 3D block: threadIdx.x, threadIdx.y, and threadIdx.z
```

### Thread Limitations

- Maximum 1024 threads per block (H100, compute capability 9.0)
- If you need more parallel tasks, use multiple blocks
- Threads within a block can synchronize with `__syncthreads()`

## 2. Blocks

### What is a Block?

- **Collection of threads** that execute cooperatively
- All threads in a block run on the **same Streaming Multiprocessor (SM)**
- Threads in a block can:
  - Share fast shared memory
  - Synchronize execution with `__syncthreads()`
  - Cooperate to solve a problem

### Block Identification

```cuda
// Built-in variables for block identification
blockIdx.x   // Block index within grid (X dimension)
blockIdx.y   // Block index within grid (Y dimension)
blockIdx.z   // Block index within grid (Z dimension)

blockDim.x   // Number of threads in block (X dimension)
blockDim.y   // Number of threads in block (Y dimension)
blockDim.z   // Number of threads in block (Z dimension)

// Examples:
gridDim.x    // Number of blocks in grid (X dimension)
gridDim.y    // Number of blocks in grid (Y dimension)
gridDim.z    // Number of blocks in grid (Z dimension)
```

### Block Dimensions

Blocks can be 1D, 2D, or 3D:

```cuda
// 1D Block: 256 threads in X dimension
dim3 blockDim(256);

// 2D Block: 16x16 = 256 threads
dim3 blockDim(16, 16);

// 3D Block: 8x8x4 = 256 threads
dim3 blockDim(8, 8, 4);
```

## 3. Grids

### What is a Grid?

- **Collection of blocks** that execute the same kernel
- Blocks can be scheduled on different SMs
- Blocks are **independent** and can run in any order
- Grid launch is one kernel invocation

### Grid Dimensions

Grids can be 1D, 2D, or 3D:

```cuda
// 1D Grid: 32 blocks
dim3 gridDim(32);

// 2D Grid: 8x8 = 64 blocks
dim3 gridDim(8, 8);

// 3D Grid: 4x4x4 = 64 blocks
dim3 gridDim(4, 4, 4);
```

## 4. Kernel Launch Syntax

```cuda
kernel<<<gridDim, blockDim>>>(arguments);

// Parameters:
// gridDim   - Number of blocks in the grid
// blockDim  - Number of threads per block
// Optional: shared memory size and stream

// Examples:
kernel<<<32, 256>>>(data);           // 32 blocks, 256 threads each
kernel<<<dim3(8,8), dim3(16,16)>>>(data);  // 2D grid and blocks
```

## Practical Examples

### Example 1: 1D Kernel Launch

```cuda
#include <stdio.h>
#include <cuda.h>

__global__ void printIndices() {
    printf("Block (%d) Thread (%d)\n", blockIdx.x, threadIdx.x);
}

int main() {
    // Launch kernel with 2 blocks, 4 threads per block
    printIndices<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}

// Output:
// Block (0) Thread (0)
// Block (0) Thread (1)
// Block (0) Thread (2)
// Block (0) Thread (3)
// Block (1) Thread (0)
// Block (1) Thread (1)
// Block (1) Thread (2)
// Block (1) Thread (3)
```

### Example 2: 2D Block Configuration

```cuda
__global__ void kernel2D() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    printf("Position (%d, %d)\n", x, y);
}

int main() {
    dim3 blockDim(4, 4);    // 4x4 = 16 threads
    dim3 gridDim(2, 2);     // 2x2 = 4 blocks
    
    kernel2D<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### Example 3: Processing Array with Multiple Blocks

```cuda
__global__ void processArray(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * 2;
    }
}

int main() {
    int n = 1000;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;  // ceil(n / blockSize)
    
    int *d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    
    processArray<<<numBlocks, blockSize>>>(d_data, n);
    
    cudaDeviceSynchronize();
    cudaFree(d_data);
    return 0;
}
```

## Understanding Warps

### What is a Warp?

- A warp is a **unit of 32 threads** scheduled for execution together
- All 32 threads in a warp execute the **same instruction** (SIMD)
- Warps are the basic scheduling unit on NVIDIA GPUs
- A block is divided into warps: `numWarps = (blockDim.x * blockDim.y * blockDim.z + 31) / 32`

### Example Warp Division

```
Block with 256 threads = 256 / 32 = 8 warps
Block with 512 threads = 512 / 32 = 16 warps (but max is 1024 per block)
```

### Warp Divergence Problem

```cuda
__global__ void divergenceExample(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // GOOD: No divergence within a warp
    if (idx % 2 == 0) {
        data[idx] = data[idx] * 2;
    } else {
        data[idx] = data[idx] + 1;
    }
    
    // BAD: Divergence - threads in same warp take different paths
    if (threadIdx.x < 16) {
        data[idx] = data[idx] * 2;
    } else {
        data[idx] = data[idx] + 1;
    }
}
```

## Key Design Principles

1. **Use multiple blocks** if you have more work than 1024 threads
2. **Keep threads per block as a multiple of 32** (warp size)
3. **Common block sizes**: 128, 256, 512 threads
4. **Minimize warp divergence** for better performance
5. **Use shared memory** for cooperation between threads in a block

## Summary Table

| Concept | Purpose | Limit | Access |
|---------|---------|-------|--------|
| **Thread** | Execute kernel code | 1024/block | threadIdx |
| **Block** | Synchronize & share memory | gridDim | blockIdx, blockDim |
| **Grid** | Launch kernel | Device limit | gridDim |
| **Warp** | Schedule execution | 32 threads | N/A (automatic) |

## Next Steps

- [Learn Indexing and Memory Access Patterns](03-Indexing-Memory.md)
- [See practical examples in code](../src/)
