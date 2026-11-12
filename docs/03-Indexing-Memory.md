# Indexing and Memory Access: Computing Global Indices

## Understanding Global Index Calculation

The most critical skill in CUDA programming is correctly computing the global index of a thread. This determines which data element each thread processes.

## 1D Indexing Formula

### Basic Formula

```
globalIdx = blockIdx.x * blockDim.x + threadIdx.x
```

### Breakdown

- **blockIdx.x**: Which block is this thread in? (0, 1, 2, ...)
- **blockDim.x**: How many threads are in each block?
- **threadIdx.x**: Which thread am I within my block? (0, 1, 2, ..., blockDim.x-1)

### Visual Example

```
Suppose:
- gridDim.x = 3 blocks
- blockDim.x = 4 threads per block

Grid Layout:
┌─────────────────────────────────────────┐
│ Block 0        │ Block 1        │ Block 2        │
│ T0 T1 T2 T3    │ T0 T1 T2 T3    │ T0 T1 T2 T3    │
└─────────────────────────────────────────┘
Global indices:
0  1  2  3      4  5  6  7      8  9  10 11

Calculation Examples:
- Thread in Block 0, Thread 0: 0 * 4 + 0 = 0
- Thread in Block 0, Thread 3: 0 * 4 + 3 = 3
- Thread in Block 1, Thread 0: 1 * 4 + 0 = 4
- Thread in Block 1, Thread 2: 1 * 4 + 2 = 6
- Thread in Block 2, Thread 3: 2 * 4 + 3 = 11
```

### Code Example

```cuda
__global__ void process1D(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check: ensure we don't go out of bounds
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    int n = 1000;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;  // Ceiling division
    
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    
    process1D<<<gridSize, blockSize>>>(d_data, n);
    
    cudaFree(d_data);
    return 0;
}
```

## 2D Indexing Formulas

### For Row-Major Layout (C/C++ style)

```
globalRow = blockIdx.y * blockDim.y + threadIdx.y
globalCol = blockIdx.x * blockDim.x + threadIdx.x
globalIdx = globalRow * cols + globalCol
```

### Visual Example

```
Matrix with 8 rows, 8 columns
Block configuration: gridDim(2,2), blockDim(4,4)

    Col: 0  1  2  3    4  5  6  7
Row 0:   [0  1  2  3 | 4  5  6  7]
Row 1:   [8  9 10 11 |12 13 14 15]
Row 2:  [16 17 18 19 |20 21 22 23]
Row 3:  [24 25 26 27 |28 29 30 31]
        ──────────────┼──────────────
Row 4:  [32 33 34 35 |36 37 38 39]
Row 5:  [40 41 42 43 |44 45 46 47]
Row 6:  [48 49 50 51 |52 53 54 55]
Row 7:  [56 57 58 59 |60 61 62 63]
        Block(0,0)  | Block(1,0)
        ──────────────┼──────────────
        Block(0,1)    | Block(1,1)

Example: Thread at Block(0,0), threadIdx(2,1)
- globalRow = 0 * 4 + 1 = 1
- globalCol = 0 * 4 + 2 = 2
- globalIdx = 1 * 8 + 2 = 10 ✓
```

### Code Example

```cuda
__global__ void process2D(float *matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        matrix[idx] = matrix[idx] * 2.0f;
    }
}

int main() {
    int rows = 1024, cols = 1024;
    int blockSize = 16;  // 16x16 = 256 threads
    
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(
        (cols + blockSize - 1) / blockSize,
        (rows + blockSize - 1) / blockSize
    );
    
    float *d_matrix;
    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    
    process2D<<<gridDim, blockDim>>>(d_matrix, rows, cols);
    
    cudaFree(d_matrix);
    return 0;
}
```

## 3D Indexing Formulas

### For 3D Array (Row-Major Order)

```
globalX = blockIdx.x * blockDim.x + threadIdx.x
globalY = blockIdx.y * blockDim.y + threadIdx.y
globalZ = blockIdx.z * blockDim.z + threadIdx.z
globalIdx = globalZ * (rows * cols) + globalY * cols + globalX
```

### Code Example

```cuda
__global__ void process3D(float *volume, int depth, int rows, int cols) {
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (z < depth && y < rows && x < cols) {
        int idx = z * (rows * cols) + y * cols + x;
        volume[idx] = volume[idx] * 2.0f;
    }
}

int main() {
    int depth = 64, rows = 128, cols = 128;
    int blockSize = 8;  // 8x8x8 = 512 threads
    
    dim3 blockDim(blockSize, blockSize, blockSize);
    dim3 gridDim(
        (cols + blockSize - 1) / blockSize,
        (rows + blockSize - 1) / blockSize,
        (depth + blockSize - 1) / blockSize
    );
    
    float *d_volume;
    cudaMalloc(&d_volume, depth * rows * cols * sizeof(float));
    
    process3D<<<gridDim, blockDim>>>(d_volume, depth, rows, cols);
    
    cudaFree(d_volume);
    return 0;
}
```

## Memory Access Patterns

### Good Pattern: Coalesced Access

```cuda
// GOOD: Sequential threads access sequential memory
// Thread 0 → Element 0, Thread 1 → Element 1, ...
__global__ void goodCoalescing(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;  // Coalesced!
}
```

### Bad Pattern: Non-Coalesced Access

```cuda
// BAD: Sequential threads access strided memory
// Thread 0 → Element 0, Thread 1 → Element 1000, ...
__global__ void badCoalescing(float *data, int stride) {
    int idx = threadIdx.x * stride;  // Large stride!
    data[idx] = data[idx] * 2.0f;   // Not coalesced, slower
}
```

## Memory Organization: Row-Major vs Column-Major

### Row-Major (C/C++, CUDA standard)

```
Matrix [3x4]:
1  2  3  4
5  6  7  8
9 10 11 12

Memory layout: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                 ^^^^^^^^^^^^^^^^
                 Row 0

Access: element[row][col] = element[row * cols + col]
```

### Column-Major (Fortran)

```
Matrix [3x4]:
1  2  3  4
5  6  7  8
9 10 11 12

Memory layout: [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
                ^^^^^^^^
                Col 0

Access: element[row][col] = element[row + col * rows]
```

## Shared Memory Indexing

### 1D Shared Memory

```cuda
__global__ void sharedMemory1D(float *global_data) {
    __shared__ float shared_data[256];
    
    int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int lIdx = threadIdx.x;  // Local index within block
    
    // Load from global to shared
    shared_data[lIdx] = global_data[gIdx];
    __syncthreads();  // Wait for all threads
    
    // Process shared memory
    shared_data[lIdx] = shared_data[lIdx] * 2.0f;
    __syncthreads();
    
    // Write back to global
    global_data[gIdx] = shared_data[lIdx];
}
```

### 2D Shared Memory with Bank Conflicts

```cuda
__global__ void shared2D(float *global_data, int N) {
    __shared__ float shared_data[32][32];  // Potential bank conflicts
    
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    
    // Load to shared (potential conflict if stride = 32)
    shared_data[ly][lx] = global_data[gy * N + gx];
    __syncthreads();
    
    // Use shared memory
    float value = shared_data[ly][lx];
}
```

## Summary: Index Calculation Quick Reference

| Dimension | Formula |
|-----------|---------|
| **1D** | `idx = blockIdx.x * blockDim.x + threadIdx.x` |
| **2D Row** | `idx = (blockIdx.y * blockDim.y + threadIdx.y) * cols + (blockIdx.x * blockDim.x + threadIdx.x)` |
| **2D Col** | `idx = (blockIdx.x * blockDim.x + threadIdx.x) * rows + (blockIdx.y * blockDim.y + threadIdx.y)` |
| **3D** | `idx = z * (rows*cols) + y * cols + x` (where z, y, x computed similarly) |

## Best Practices

1. **Always check bounds**: `if (idx < n) {...}`
2. **Use coalesced memory access**: Sequential threads access sequential memory
3. **Avoid warp divergence**: All threads in a warp should follow the same path
4. **Understand your data layout**: Row-major vs Column-major affects indexing
5. **Use shared memory for cooperation**: Reduces global memory bandwidth
6. **Minimize bank conflicts**: Avoid stride-32 patterns in shared memory

## Next Steps

- [SLES15 SP7 Installation Guide](04-SLES15-Setup.md)
- [Sample Code Examples](../src/)
