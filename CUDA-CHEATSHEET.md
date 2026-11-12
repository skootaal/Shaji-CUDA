# CUDA Programming Cheat Sheet

A quick reference guide for CUDA programming concepts and commonly used code patterns.

## Table of Contents

1. [Memory Management](#memory-management)
2. [Kernel Launching](#kernel-launching)
3. [Thread Indexing](#thread-indexing)
4. [Synchronization](#synchronization)
5. [Memory Hierarchy](#memory-hierarchy)
6. [Error Handling](#error-handling)
7. [Performance Tips](#performance-tips)
8. [Common Patterns](#common-patterns)

---

## Memory Management

### Allocating GPU Memory

```cuda
// Allocate device memory
float *d_data;
cudaMalloc((void**)&d_data, size_bytes);

// Allocate pinned (page-locked) host memory (faster transfers)
float *h_data;
cudaMallocHost((void**)&h_data, size_bytes);

// Allocate unified memory (accessible by both CPU and GPU)
float *ud_data;
cudaMallocManaged((void**)&ud_data, size_bytes);
```

### Copying Data

```cuda
// Host to Device
cudaMemcpy(d_data, h_data, size_bytes, cudaMemcpyHostToDevice);

// Device to Host
cudaMemcpy(h_data, d_data, size_bytes, cudaMemcpyDeviceToHost);

// Device to Device
cudaMemcpy(d_data2, d_data1, size_bytes, cudaMemcpyDeviceToDevice);

// Async copy (non-blocking)
cudaMemcpyAsync(d_data, h_data, size_bytes, cudaMemcpyHostToDevice, stream);
```

### Freeing Memory

```cuda
cudaFree(d_data);
cudaFreeHost(h_data);
cudaFreeManaged(ud_data);
```

### Memory Set

```cuda
// Set device memory to a value
cudaMemset(d_data, 0, size_bytes);
```

---

## Kernel Launching

### Basic Syntax

```cuda
// Define kernel
__global__ void kernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx;
}

// Launch kernel
kernel<<<gridDim, blockDim>>>(data);

// With stream and shared memory
kernel<<<gridDim, blockDim, shared_memory_bytes, stream>>>(data);
```

### Grid and Block Configuration

```cuda
// 1D Configuration
int threads_per_block = 256;
int num_blocks = (n + threads_per_block - 1) / threads_per_block;
kernel<<<num_blocks, threads_per_block>>>(data, n);

// 2D Configuration
dim3 blockDim(16, 16);  // 16x16 = 256 threads
dim3 gridDim(
    (width + blockDim.x - 1) / blockDim.x,
    (height + blockDim.y - 1) / blockDim.y
);
kernel<<<gridDim, blockDim>>>(data, width, height);

// 3D Configuration
dim3 blockDim(8, 8, 4);  // 8x8x4 = 256 threads
dim3 gridDim(
    (depth_x + blockDim.x - 1) / blockDim.x,
    (depth_y + blockDim.y - 1) / blockDim.y,
    (depth_z + blockDim.z - 1) / blockDim.z
);
kernel<<<gridDim, blockDim>>>(data);
```

---

## Thread Indexing

### 1D Index Calculation

```cuda
__global__ void kernel1D(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

### 2D Index Calculation

```cuda
__global__ void kernel2D(float *matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;  // Row-major layout
        matrix[idx] = matrix[idx] * 2.0f;
    }
}
```

### 3D Index Calculation

```cuda
__global__ void kernel3D(float *volume, int depth, int height, int width) {
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (z < depth && y < height && x < width) {
        int idx = z * (height * width) + y * width + x;
        volume[idx] = volume[idx] * 2.0f;
    }
}
```

### Built-in Variables

```cuda
// Thread indices
threadIdx.x, threadIdx.y, threadIdx.z

// Block indices
blockIdx.x, blockIdx.y, blockIdx.z

// Grid/Block dimensions
blockDim.x, blockDim.y, blockDim.z
gridDim.x, gridDim.y, gridDim.z

// Warp-related
warpSize  // 32 for most NVIDIA GPUs
```

---

## Synchronization

### Synchronizing Threads in a Block

```cuda
__global__ void kernel_with_sync(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Do work
    data[idx] = data[idx] * 2.0f;
    
    // Wait for all threads in block to reach this point
    __syncthreads();
    
    // More work after sync
    data[idx] = data[idx] + 1.0f;
}
```

### Synchronizing Host and Device

```cuda
// Wait for all kernels to complete
cudaDeviceSynchronize();

// Wait for a specific stream
cudaStreamSynchronize(stream);
```

### Warp-level Synchronization (Compute Capability 7.0+)

```cuda
// Synchronize all threads in a warp
__syncwarp();

// Synchronize threads in a warp with mask
__syncwarp(0xFFFFFFFF);  // All 32 threads
```

---

## Memory Hierarchy

### Global Memory

```cuda
// Slow but large (GB)
__global__ void kernel(float *global_data) {
    global_data[threadIdx.x] = 1.0f;
}
```

### Shared Memory

```cuda
// Fast but small (KB)
__global__ void kernel_shared(float *global_data) {
    __shared__ float shared_data[256];  // 1KB per 4 bytes
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // Load to shared
    shared_data[local_idx] = global_data[idx];
    __syncthreads();
    
    // Use shared data (much faster!)
    float value = shared_data[local_idx];
    
    // Store result
    global_data[idx] = value * 2.0f;
}
```

### Local Memory

```cuda
// Per-thread memory (registers or global)
__global__ void kernel_local(float *data) {
    float local_var = 1.0f;  // In register if possible
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    data[idx] = local_var;
}
```

### Constant Memory

```cuda
// Read-only, very fast
__constant__ float constant_data[1024];

__global__ void kernel_const(float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = constant_data[idx];
}

// Set constant memory from host
float host_const[1024];
cudaMemcpyToSymbol(constant_data, host_const, sizeof(host_const));
```

### Texture Memory

```cuda
// Read-optimized memory with caching
texture<float, 1, cudaReadModeElementType> tex_ref;

__global__ void kernel_tex(float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = tex1Dfetch(tex_ref, idx);
}

// Bind texture from host
cudaBindTexture(NULL, tex_ref, device_data, size_bytes);
```

---

## Error Handling

### Checking Errors

```cuda
// Check kernel launch errors
kernel<<<blocks, threads>>>(data);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
}

// Check CUDA API errors
err = cudaMalloc(&d_data, size_bytes);
if (err != cudaSuccess) {
    fprintf(stderr, "Malloc failed: %s\n", cudaGetErrorString(err));
}

// Synchronize and check for kernel errors
cudaDeviceSynchronize();
err = cudaGetLastError();
```

### Error Macro

```cuda
#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Usage
CHECK_CUDA(cudaMalloc(&d_data, size_bytes));
CHECK_CUDA(cudaMemcpy(d_data, h_data, size_bytes, cudaMemcpyHostToDevice));
```

---

## Performance Tips

### Memory Coalescing

```cuda
// GOOD: Sequential access
__global__ void good_coalesce(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;  // Coalesced!
}

// BAD: Strided access
__global__ void bad_coalesce(float *data, int stride) {
    int idx = threadIdx.x * stride;  // Non-coalesced
    data[idx] = data[idx] * 2.0f;
}
```

### Avoiding Warp Divergence

```cuda
// GOOD: No divergence within warp
__global__ void no_divergence(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 2 == 0) {
        data[idx] = data[idx] * 2;
    }
}

// BAD: Divergence within warp
__global__ void has_divergence(int *data) {
    if (threadIdx.x < 16) {
        data[threadIdx.x] = 0;
    }
    // Different path for threads 16-31
}
```

### Occupancy

```cuda
// Higher occupancy = better GPU utilization
// Aim for 2-4 blocks per SM
// With 256 threads/block: need 512-1024 threads available per SM

int block_size = 256;  // Good choice
// int block_size = 32;  // Too small, low occupancy
// int block_size = 1024; // Maximum, but reduces blocks per SM
```

---

## Common Patterns

### Reduction (Sum)

```cuda
__global__ void reduce_sum(float *data, float *result, int n) {
    __shared__ float partial_sum[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    partial_sum[local_idx] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_idx < s) {
            partial_sum[local_idx] += partial_sum[local_idx + s];
        }
        __syncthreads();
    }
    
    if (local_idx == 0) {
        atomicAdd(result, partial_sum[0]);
    }
}
```

### Scan (Prefix Sum)

```cuda
__global__ void scan_inclusive(float *data, int n) {
    __shared__ float temp[512];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    temp[local_idx] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    // Up-sweep phase
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (local_idx >= offset) {
            temp[local_idx] += temp[local_idx - offset];
        }
        __syncthreads();
    }
    
    if (idx < n) {
        data[idx] = temp[local_idx];
    }
}
```

### Atomic Operations

```cuda
__global__ void atomic_example(int *counter, int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Atomic increment
    atomicAdd(counter, 1);
    
    // Atomic min/max
    atomicMin(counter, data[idx]);
    atomicMax(counter, data[idx]);
    
    // Atomic exchange
    atomicExch(counter, data[idx]);
}
```

### Histogram

```cuda
__global__ void histogram(int *data, int *hist, int n, int bins) {
    __shared__ int shared_hist[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // Initialize shared histogram
    if (local_idx < bins) {
        shared_hist[local_idx] = 0;
    }
    __syncthreads();
    
    // Accumulate in shared memory
    if (idx < n) {
        int bin = min(data[idx], bins - 1);
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();
    
    // Write to global memory
    if (local_idx < bins) {
        atomicAdd(&hist[local_idx], shared_hist[local_idx]);
    }
}
```

---

## Quick Reference Table

| Operation | Code | Notes |
|-----------|------|-------|
| Allocate GPU memory | `cudaMalloc(&ptr, size)` | Returns error code |
| Copy H→D | `cudaMemcpy(d, h, size, cudaMemcpyHostToDevice)` | Blocking |
| Copy D→H | `cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost)` | Blocking |
| Free GPU memory | `cudaFree(ptr)` | Always clean up |
| Launch kernel | `kernel<<<blocks, threads>>>(args)` | Asynchronous |
| Sync device | `cudaDeviceSynchronize()` | Wait for GPU |
| Get last error | `cudaGetLastError()` | Check for errors |
| 1D index | `blockIdx.x * blockDim.x + threadIdx.x` | Standard formula |
| Shared memory | `__shared__ float data[256]` | Per-block allocation |
| Sync threads | `__syncthreads()` | Block-level sync |
| Atomic add | `atomicAdd(ptr, value)` | Thread-safe |

---

## Compilation Flags

```bash
# Basic compilation
nvcc -o program program.cu

# With optimization
nvcc -O3 -arch=sm_90 -o program program.cu

# With debug info
nvcc -g -G -o program program.cu

# With libraries
nvcc -o program program.cu -lcublas -lcufft

# Specific compute capability
nvcc -arch=sm_90 -o program program.cu  # H100
nvcc -arch=sm_80 -o program program.cu  # A100
nvcc -arch=sm_89 -o program program.cu  # RTX 40-series

# Multiple architectures (fat binary)
nvcc -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -o program program.cu
```

---

**Keep this handy while coding CUDA programs!** 🚀
