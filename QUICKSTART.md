# Quick Start Guide: Getting Started with Shaji-CUDA

This guide will help you get up and running with CUDA programming in 5 minutes!

## Prerequisites

- ✓ NVIDIA GPU (H100, A100, RTX 4090, or similar)
- ✓ CUDA Toolkit 12.x installed
- ✓ NVIDIA drivers installed
- ✓ Linux (SLES15 SP7, Ubuntu, or similar)

**Not installed yet?** See [SLES15 SP7 Setup Guide](docs/04-SLES15-Setup.md)

## Step 1: Verify Your CUDA Installation

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output: GPU information with driver version

# Check CUDA compiler
nvcc --version

# Expected output: CUDA compilation tools version
```

If either command fails, see [SLES15 Setup Guide](docs/04-SLES15-Setup.md) for troubleshooting.

## Step 2: Clone This Repository

```bash
git clone https://github.com/skootaal/Shaji-CUDA.git
cd Shaji-CUDA
```

## Step 3: Compile All Programs

### Using Makefile (Recommended)

```bash
cd src
make all

# Output:
# Compiling vector_add...
#   ✓ bin/vector_add compiled successfully
# Compiling matrix_multiply...
#   ✓ bin/matrix_multiply compiled successfully
# Compiling h100_test...
#   ✓ bin/h100_test compiled successfully
# Build complete!
```

### Or Compile Individually

```bash
# Vector addition
nvcc -O2 -arch=sm_90 -o bin/vector_add src/vector_add.cu

# Matrix multiplication (requires cuBLAS)
nvcc -O2 -arch=sm_90 -o bin/matrix_multiply src/matrix_multiply.cu -lcublas

# H100 performance test
nvcc -O2 -arch=sm_90 -o bin/h100_test src/h100_test.cu
```

## Step 4: Run the Programs

### Run Vector Addition

```bash
cd src
./bin/vector_add

# Output:
# Vector Addition on GPU
# ========================
# Number of elements: 50000
# Memory required: 0.20 MB
# 
# Initializing vectors...
# GPU memory allocated successfully
# Copying data from CPU to GPU...
# 
# Kernel Configuration:
#   Threads per block: 256
#   Blocks per grid:   196
#   Total threads:     50176
# 
# Launching kernel...
# Kernel execution time: 0.123 ms
# Copying results from GPU to CPU...
# Verifying results...
# ✓ All results verified successfully!
```

### Run Matrix Multiplication

```bash
./bin/matrix_multiply

# Output:
# Matrix Multiplication on GPU (CUDA)
# ====================================
# Matrix dimensions: C[1024 x 1024] = A[1024 x 1024] * B[1024 x 1024]
# 
# --- Method 1: Naive CUDA Kernel ---
# Naive kernel time: 4.567 ms
# 
# --- Method 2: Optimized CUDA Kernel (Shared Memory) ---
# Shared memory kernel time: 1.234 ms
# 
# --- Method 3: cuBLAS Library ---
# cuBLAS kernel time: 0.789 ms
# 
# --- Performance Summary ---
# Naive kernel:         225.00 GFLOPS
# Shared memory kernel: 836.13 GFLOPS (3.71x faster)
# cuBLAS kernel:        1,249.46 GFLOPS (1.49x faster)
```

### Run H100 Performance Test

```bash
./bin/h100_test

# Output:
# ========================================
# GPU Information
# ========================================
# Device Name: NVIDIA H100 PCIe
# Compute Capability: 9.0
# Total Global Memory: 80.00 GB
# ...
# 
# H100 Performance Test Suite
# ============================
# 
# TEST 1: FP32 (Single Precision) Compute Performance
# Average execution time: 15.234 ms
# Peak FP32 throughput: 2,145.67 TFLOPS
# (Theoretical peak for H100: 2,660 TFLOPS)
# 
# TEST 2: FP64 (Double Precision) Compute Performance
# Average execution time: 30.456 ms
# Peak FP64 throughput: 1,072.83 TFLOPS
# 
# TEST 3: Global Memory Read Bandwidth
# Memory bandwidth: 3,200.45 GB/s
# 
# TEST 4: Shared Memory Bandwidth
# Shared memory bandwidth: 32,456.78 GB/s
# 
# TEST 5: GPU-CPU PCIe Bandwidth
# Host to Device bandwidth: 64.32 GB/s
# Device to Host bandwidth: 62.18 GB/s
```

## Step 5: Learn CUDA Concepts

Read the documentation in order:

1. **[CUDA Basics](docs/01-CUDA-Basics.md)** - Fundamental concepts
   - CPU vs GPU
   - Memory hierarchy
   - Execution model overview

2. **[Kernels, Blocks, Threads](docs/02-Kernels-Blocks-Threads.md)** - Execution model
   - Thread identification
   - Block organization
   - Grid configuration
   - Warp execution

3. **[Indexing and Memory](docs/03-Indexing-Memory.md)** - Index calculations
   - 1D, 2D, 3D indexing formulas with examples
   - Memory access patterns
   - Coalescing
   - Shared memory

4. **[SLES15 SP7 Setup](docs/04-SLES15-Setup.md)** - Installation guide
   - Driver installation
   - CUDA Toolkit setup
   - Environment configuration

5. **[Build and Execute](docs/05-Build-Execute.md)** - Compilation
   - nvcc compiler flags
   - Makefile examples
   - CMake setup
   - Profiling and debugging

## Makefile Quick Reference

```bash
cd src

# Build all programs
make all

# Build and run vector addition
make run_vector_add

# Build and run matrix multiplication
make run_matrix

# Build and run H100 test
make run_h100

# Build and run all
make run_all

# Build with debug symbols
make debug

# Clean build artifacts
make clean

# Show help
make help
```

## Common Tasks

### Compile a Single File

```bash
nvcc -O2 -arch=sm_90 -o program program.cu
```

### Compile with Optimization

```bash
nvcc -O3 -arch=sm_90 -o program program.cu
```

### Compile with Debug Info

```bash
nvcc -g -G -arch=sm_90 -o program program.cu
```

### Profile Execution

```bash
nsys profile ./program
nsys stats report.nsys-rep
```

### Run with GPU Monitoring

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Run program
./program
```

## Troubleshooting

### "nvcc: command not found"
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### "error: cannot find -lcublas"
```bash
# Specify library path
nvcc -L/usr/local/cuda/lib64 -o program program.cu -lcublas
```

### "CUDA kernel not found"
Check that your kernel is declared with `__global__`

### Segmentation fault
Compile with debug symbols and use cuda-gdb:
```bash
nvcc -g -G -o program program.cu
cuda-gdb ./program
```

## Next Steps

1. **Modify the sample code** - Try changing parameters in vector_add.cu
2. **Write your own kernel** - Implement a different algorithm
3. **Profile your code** - Use nsys to find bottlenecks
4. **Optimize for performance** - Use shared memory and coalescing
5. **Explore CUDA libraries** - Use cuBLAS, cuFFT, TensorRT

## Resources

- **Official NVIDIA CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Samples**: `/usr/local/cuda/samples/`
- **H100 Specs**: https://www.nvidia.com/en-us/data-center/h100/

## File Structure

```
Shaji-CUDA/
├── README.md                          # Main README
├── QUICKSTART.md                      # This file
├── docs/
│   ├── 01-CUDA-Basics.md
│   ├── 02-Kernels-Blocks-Threads.md
│   ├── 03-Indexing-Memory.md
│   ├── 04-SLES15-Setup.md
│   └── 05-Build-Execute.md
└── src/
    ├── Makefile
    ├── vector_add.cu
    ├── matrix_multiply.cu
    └── h100_test.cu
```

## Performance Tips

1. **Use appropriate block size** - 256 or 512 threads per block
2. **Maximize occupancy** - Keep SMs busy with multiple blocks
3. **Coalesce memory access** - Threads access sequential memory
4. **Use shared memory** - For cooperative access between threads
5. **Minimize warp divergence** - All threads in warp take same path
6. **Batch transfers** - Reduce PCIe overhead

## Contact & Support

For issues or questions:
- Check the [documentation](docs/)
- Review the [sample code](src/)
- Consult NVIDIA's [official guides](https://docs.nvidia.com/cuda/)

Happy CUDA programming! 🚀
