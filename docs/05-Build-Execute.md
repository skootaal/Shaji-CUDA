# Building and Executing CUDA Programs

## Overview

This guide covers how to compile CUDA programs using `nvcc`, the NVIDIA CUDA compiler, and execute them on your GPU.

## CUDA Compilation Basics

### The nvcc Compiler

`nvcc` (NVIDIA CUDA Compiler) separates code into host (CPU) and device (GPU) code:

```
Source Code (.cu)
       ↓
nvcc Compiler
   ├─→ CPU Code (passed to host compiler: gcc/g++)
   └─→ GPU Code (compiled to PTX, then to machine code)
       ↓
   Executable
```

### Basic Compilation Command

```bash
nvcc -o output_name input.cu
```

### Common nvcc Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `-o` | Output file name | `nvcc -o myprogram program.cu` |
| `-arch=sm_90` | Target GPU architecture | `nvcc -arch=sm_90 program.cu` |
| `-O2` | Optimization level | `nvcc -O2 program.cu` |
| `-g` | Debug symbols | `nvcc -g program.cu` |
| `-std=c++17` | C++ standard | `nvcc -std=c++17 program.cu` |
| `-I` | Include directory | `nvcc -I/path/to/include program.cu` |
| `-L` | Library directory | `nvcc -L/path/to/lib program.cu` |
| `-l` | Link library | `nvcc -lcublas program.cu` |
| `--device-debug` | Device-side debugging | `nvcc --device-debug program.cu` |

## Compilation Examples

### Example 1: Basic Compilation

```bash
# Simple compilation
nvcc -o vector_add vector_add.cu

# With optimization
nvcc -O2 -o vector_add vector_add.cu

# For H100 (Compute Capability 9.0)
nvcc -arch=sm_90 -O2 -o vector_add vector_add.cu
```

### Example 2: Compilation with Libraries

```bash
# Compile with cuBLAS
nvcc -o matrix_mult matrix_multiply.cu -lcublas

# Compile with cuFFT
nvcc -o fft_test fft_test.cu -lcufft

# Compile with multiple libraries
nvcc -o complex_app complex.cu -lcublas -lcufft -lcurand
```

### Example 3: Compilation with Custom Paths

```bash
# If CUDA is not in standard location
nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
     -o program program.cu -lcudart

# With custom include directories
nvcc -I./include -I/usr/local/cuda/include \
     -o program program.cu
```

### Example 4: Debug Compilation

```bash
# Compile with debugging symbols
nvcc -g -G -o program_debug program.cu

# Can use cuda-gdb for debugging
cuda-gdb ./program_debug
```

## Makefile for CUDA Projects

### Simple Makefile

```makefile
# Makefile for CUDA projects

NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_90
CC_FLAGS = -std=c++17

# Target rules
all: vector_add matrix_multiply h100_test

vector_add: src/vector_add.cu
	$(NVCC) $(NVCC_FLAGS) -o bin/vector_add src/vector_add.cu

matrix_multiply: src/matrix_multiply.cu
	$(NVCC) $(NVCC_FLAGS) -o bin/matrix_multiply src/matrix_multiply.cu

h100_test: src/h100_test.cu
	$(NVCC) $(NVCC_FLAGS) -o bin/h100_test src/h100_test.cu

clean:
	rm -f bin/vector_add bin/matrix_multiply bin/h100_test

.PHONY: all clean
```

### Advanced Makefile with Libraries

```makefile
# Advanced Makefile with cuBLAS, cuFFT support

NVCC = nvcc
CUDA_PATH = /usr/local/cuda
NVCC_FLAGS = -O2 -arch=sm_90 -std=c++17

# Include and library paths
INCLUDES = -I$(CUDA_PATH)/include -I./include
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcufft

# Source and object files
SOURCES = src/vector_add.cu src/matrix_multiply.cu
OBJECTS = $(SOURCES:.cu=.o)
TARGETS = bin/vector_add bin/matrix_multiply

# Compilation rules
all: $(TARGETS)

bin/%: src/%.cu
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

src/%.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGETS)

run_vector_add: bin/vector_add
	./bin/vector_add

run_matrix: bin/matrix_multiply
	./bin/matrix_multiply

.PHONY: all clean run_vector_add run_matrix
```

## Execution

### Basic Execution

```bash
# Run compiled program
./vector_add

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 ./vector_add

# Run multiple times
for i in {1..5}; do ./vector_add; done
```

### Monitoring Execution

```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# In another terminal, run your program
./h100_test

# Check detailed GPU statistics
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

### Profiling and Optimization

```bash
# Profile with NVIDIA Profiler
nsys profile ./h100_test

# Analyze results
nsys stats report.nsys-rep

# Use NVIDIA Visual Profiler (GUI)
nvvp ./h100_test &
```

## Output Redirection and Logging

### Save Output to File

```bash
# Redirect stdout
./vector_add > output.log

# Redirect stderr
./vector_add 2> errors.log

# Redirect both
./vector_add > output.log 2>&1

# Append to file
./vector_add >> output.log 2>&1
```

### Timestamped Execution Log

```bash
# Log with timestamp
echo "[$(date)]" >> execution.log
./vector_add >> execution.log 2>&1
echo "Exit code: $?" >> execution.log
```

## CMake for CUDA Projects

### CMakeLists.txt Example

```cmake
cmake_minimum_required(VERSION 3.18)
project(CUDAProject CUDA CXX)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add executable
add_executable(vector_add src/vector_add.cu)
add_executable(matrix_multiply src/matrix_multiply.cu)
add_executable(h100_test src/h100_test.cu)

# Link CUDA libraries
target_link_libraries(vector_add CUDA::cudart)
target_link_libraries(matrix_multiply CUDA::cudart CUDA::cublas)
target_link_libraries(h100_test CUDA::cudart)

# Set CUDA architecture
set_target_properties(vector_add PROPERTIES CUDA_ARCHITECTURES 90)
set_target_properties(matrix_multiply PROPERTIES CUDA_ARCHITECTURES 90)
set_target_properties(h100_test PROPERTIES CUDA_ARCHITECTURES 90)
```

### Build with CMake

```bash
# Create build directory
mkdir build
cd build

# Generate build files
cmake ..

# Compile
make

# Run
./vector_add
./matrix_multiply
./h100_test
```

## Performance Measurement

### Timing Code Execution

```bash
# Using Linux time command
time ./h100_test

# Output shows:
# real    0m1.234s  (wall clock time)
# user    0m0.056s  (CPU time)
# sys     0m0.012s  (system time)
```

### CUDA Event Timing (In Code)

```cuda
#include <stdio.h>
#include <cuda.h>

int main() {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    // Your CUDA kernel here
    kernel<<<blocks, threads>>>();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Execution time: %.3f ms\n", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
```

## Troubleshooting Compilation

### Issue: "nvcc: command not found"

```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

### Issue: "error: cannot find -lcublas"

```bash
# Specify library path
nvcc -L/usr/local/cuda/lib64 -o program program.cu -lcublas

# Or update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: "error: no kernels found"

```bash
# Ensure __global__ functions are present
# Check that function is defined before kernel launch
```

### Issue: "Segmentation fault" at runtime

```bash
# Compile with debug info
nvcc -g -G -o program program.cu

# Run with debugger
cuda-gdb ./program

# Check for invalid memory access
# Common issues:
# - Out of bounds array access
# - Uninitialized device memory
# - Missing __syncthreads()
```

## Build and Run Scripts

### Complete Build Script

```bash
#!/bin/bash
# build.sh - Compile all CUDA programs

set -e  # Exit on error

NVCC=nvcc
ARCH=sm_90
FLAGS="-O2 -arch=$ARCH -std=c++17"
OUTPUT_DIR="bin"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Building CUDA programs..."

# Compile each program
echo "  Compiling vector_add..."
$NVCC $FLAGS -o $OUTPUT_DIR/vector_add src/vector_add.cu

echo "  Compiling matrix_multiply..."
$NVCC $FLAGS -o $OUTPUT_DIR/matrix_multiply src/matrix_multiply.cu -lcublas

echo "  Compiling h100_test..."
$NVCC $FLAGS -o $OUTPUT_DIR/h100_test src/h100_test.cu

echo "Build complete! Executables in $OUTPUT_DIR/"
```

### Run All Tests Script

```bash
#!/bin/bash
# run_all.sh - Execute all compiled programs

set -e

OUTPUT_DIR="bin"

echo "Running all CUDA programs..."

echo ""
echo "=== Vector Add Test ==="
$OUTPUT_DIR/vector_add

echo ""
echo "=== Matrix Multiply Test ==="
$OUTPUT_DIR/matrix_multiply

echo ""
echo "=== H100 Performance Test ==="
$OUTPUT_DIR/h100_test

echo ""
echo "All tests completed!"
```

## Quick Reference

```bash
# Compile a single file
nvcc -o program program.cu

# Compile for H100 (sm_90)
nvcc -arch=sm_90 -O2 -o program program.cu

# Compile with libraries
nvcc -o program program.cu -lcublas -lcufft

# Debug compilation
nvcc -g -G -o program program.cu

# Run program
./program

# Monitor during execution
nvidia-smi -l 1 &
./program

# Profile execution
nsys profile ./program

# Timing
time ./program
```

## Next Steps

- [Vector Addition Sample Code](../src/vector_add.cu)
- [Matrix Multiplication Sample Code](../src/matrix_multiply.cu)
- [H100 Performance Test](../src/h100_test.cu)
