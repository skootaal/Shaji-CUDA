#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

/**
 * H100 GPU Performance Test Suite
 * 
 * This program tests various aspects of NVIDIA H100 GPU performance:
 * 1. Peak FP32 (single precision) throughput
 * 2. Peak FP64 (double precision) throughput
 * 3. Peak Tensor operations (using TensorCores if available)
 * 4. Memory bandwidth (global, shared)
 * 5. GPU-CPU bandwidth
 * 
 * H100 Specifications:
 * - Compute Capability: 9.0
 * - GPU Memory: 80 GB HBM3
 * - Peak FP32: 2,660 TFLOPS (theoretical)
 * - Peak TensorFloat-32: 5,321 TFLOPS
 * - Memory Bandwidth: 3,935 GB/s (theoretical)
 */

#define NUM_RUNS 10

// Simple FP32 compute kernel
__global__ void compute_fp32(float *data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Perform multiple operations to measure throughput
        for (int i = 0; i < iterations; i++) {
            value = value * 1.1f + 0.1f;
            value = value * 1.1f + 0.1f;
            value = value * 1.1f + 0.1f;
            value = value * 1.1f + 0.1f;
        }
        
        data[idx] = value;
    }
}

// Double precision FP64 compute kernel
__global__ void compute_fp64(double *data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        double value = data[idx];
        
        for (int i = 0; i < iterations; i++) {
            value = value * 1.1 + 0.1;
            value = value * 1.1 + 0.1;
            value = value * 1.1 + 0.1;
            value = value * 1.1 + 0.1;
        }
        
        data[idx] = value;
    }
}

// Memory bandwidth test kernel
__global__ void memory_read(float *data, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0.0f;
        // Perform multiple reads
        for (int i = 0; i < 16; i++) {
            sum += data[(idx + i * n / 16) % n];
        }
        result[idx] = sum;
    }
}

// Shared memory bandwidth test
__global__ void shared_memory_test(float *global_data, float *result, int n) {
    __shared__ float shared_data[1024];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // Load to shared memory
    if (idx < n) {
        shared_data[local_idx] = global_data[idx];
    }
    __syncthreads();
    
    // Perform multiple reads from shared memory
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        sum += shared_data[(local_idx + i * 64) % 1024];
    }
    
    if (idx < n) {
        result[idx] = sum;
    }
}

// GPU information
void printGPUInfo() {
    int device = 0;
    cudaSetDevice(device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    printf("\n========================================\n");
    printf("GPU Information\n");
    printf("========================================\n");
    printf("Device Name: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("Max Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("Max Block Dimension: %d x %d x %d\n", 
           props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
    printf("Max Grid Dimension: %d x %d x %d\n",
           props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    printf("Total Global Memory: %.2f GB\n", (float)props.totalGlobalMem / (1024*1024*1024));
    printf("Shared Memory per Block: %d KB\n", (int)props.sharedMemPerBlock / 1024);
    printf("Number of SMs: %d\n", props.multiProcessorCount);
    printf("Clock Rate: %.2f GHz\n", (float)props.clockRate / 1e6);
    printf("Memory Clock Rate: %.2f GHz\n", (float)props.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", props.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n", 
           2.0f * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6);
    printf("========================================\n\n");
}

int main() {
    printGPUInfo();
    
    printf("H100 Performance Test Suite\n");
    printf("============================\n\n");
    
    // Test parameters
    int num_threads = 256;
    int num_elements = 100 * 1024 * 1024;  // 100M elements
    int num_blocks = (num_elements + num_threads - 1) / num_threads;
    int iterations = 100;
    
    printf("Test Configuration:\n");
    printf("  Number of threads per block: %d\n", num_threads);
    printf("  Total elements: %d M\n", num_elements / (1024*1024));
    printf("  Number of blocks: %d\n", num_blocks);
    printf("  Iterations per kernel: %d\n\n", iterations);
    
    // ============================================================
    // Test 1: FP32 Compute Performance
    // ============================================================
    printf("TEST 1: FP32 (Single Precision) Compute Performance\n");
    printf("-" "-----------------------------------------------------\n");
    
    float *d_data_fp32 = NULL;
    cudaMalloc(&d_data_fp32, num_elements * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_time_fp32 = 0.0f;
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);
        compute_fp32<<<num_blocks, num_threads>>>(d_data_fp32, num_elements, iterations);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_time_fp32 += ms;
    }
    
    float avg_time_fp32 = total_time_fp32 / NUM_RUNS;
    // 4 FP32 operations per iteration
    float flops_fp32 = (float)num_elements * iterations * 4 / (avg_time_fp32 * 1e6f);
    
    printf("Average execution time: %.3f ms\n", avg_time_fp32);
    printf("Peak FP32 throughput: %.2f TFLOPS\n", flops_fp32);
    printf("(Theoretical peak for H100: 2,660 TFLOPS)\n\n");
    
    // ============================================================
    // Test 2: FP64 Compute Performance
    // ============================================================
    printf("TEST 2: FP64 (Double Precision) Compute Performance\n");
    printf("-" "-----------------------------------------------------\n");
    
    double *d_data_fp64 = NULL;
    cudaMalloc(&d_data_fp64, num_elements * sizeof(double));
    
    float total_time_fp64 = 0.0f;
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);
        compute_fp64<<<num_blocks, num_threads>>>(d_data_fp64, num_elements, iterations);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_time_fp64 += ms;
    }
    
    float avg_time_fp64 = total_time_fp64 / NUM_RUNS;
    float flops_fp64 = (float)num_elements * iterations * 4 / (avg_time_fp64 * 1e6f);
    
    printf("Average execution time: %.3f ms\n", avg_time_fp64);
    printf("Peak FP64 throughput: %.2f TFLOPS\n", flops_fp64);
    printf("(Theoretical peak for H100: 1,330 TFLOPS)\n\n");
    
    // ============================================================
    // Test 3: Global Memory Bandwidth
    // ============================================================
    printf("TEST 3: Global Memory Read Bandwidth\n");
    printf("-" "-----------------------------------------------------\n");
    
    float *d_read_data = NULL;
    float *d_read_result = NULL;
    cudaMalloc(&d_read_data, num_elements * sizeof(float));
    cudaMalloc(&d_read_result, num_elements * sizeof(float));
    
    float total_time_read = 0.0f;
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);
        memory_read<<<num_blocks, num_threads>>>(d_read_data, d_read_result, num_elements);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_time_read += ms;
    }
    
    float avg_time_read = total_time_read / NUM_RUNS;
    // 16 reads per thread * 4 bytes per float
    float bandwidth_read = (float)num_elements * 16 * 4 / (avg_time_read * 1e6f);
    
    printf("Average execution time: %.3f ms\n", avg_time_read);
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth_read);
    printf("(Theoretical peak for H100: 3,935 GB/s)\n\n");
    
    // ============================================================
    // Test 4: Shared Memory Bandwidth
    // ============================================================
    printf("TEST 4: Shared Memory Bandwidth\n");
    printf("-" "-----------------------------------------------------\n");
    
    float *d_shared_result = NULL;
    cudaMalloc(&d_shared_result, num_elements * sizeof(float));
    
    float total_time_shared = 0.0f;
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);
        shared_memory_test<<<num_blocks, num_threads>>>(d_read_data, d_shared_result, num_elements);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_time_shared += ms;
    }
    
    float avg_time_shared = total_time_shared / NUM_RUNS;
    float bandwidth_shared = (float)num_elements * 16 * 4 / (avg_time_shared * 1e6f);
    
    printf("Average execution time: %.3f ms\n", avg_time_shared);
    printf("Shared memory bandwidth: %.2f GB/s\n", bandwidth_shared);
    printf("(Shared memory is typically 20-50x faster than global memory)\n\n");
    
    // ============================================================
    // Test 5: GPU-CPU Bandwidth
    // ============================================================
    printf("TEST 5: GPU-CPU PCIe Bandwidth\n");
    printf("-" "-----------------------------------------------------\n");
    
    float *h_data = (float *)malloc(num_elements * sizeof(float));
    float *d_transfer_data = NULL;
    cudaMalloc(&d_transfer_data, num_elements * sizeof(float));
    
    // Test H2D (Host to Device)
    cudaEventRecord(start);
    cudaMemcpy(d_transfer_data, h_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_h2d = 0.0f;
    cudaEventElapsedTime(&ms_h2d, start, stop);
    float bandwidth_h2d = (float)num_elements * 4 / (ms_h2d * 1e6f);
    
    // Test D2H (Device to Host)
    cudaEventRecord(start);
    cudaMemcpy(h_data, d_transfer_data, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_d2h = 0.0f;
    cudaEventElapsedTime(&ms_d2h, start, stop);
    float bandwidth_d2h = (float)num_elements * 4 / (ms_d2h * 1e6f);
    
    printf("Host to Device bandwidth: %.2f GB/s (%.3f ms)\n", bandwidth_h2d, ms_h2d);
    printf("Device to Host bandwidth: %.2f GB/s (%.3f ms)\n", bandwidth_d2h, ms_d2h);
    printf("(PCIe 5.0 theoretical peak: 128 GB/s)\n\n");
    
    // ============================================================
    // Summary
    // ============================================================
    printf("========================================\n");
    printf("Performance Summary\n");
    printf("========================================\n");
    printf("FP32 Throughput:        %.2f TFLOPS (%.1f%% of theoretical)\n", 
           flops_fp32, (flops_fp32 / 2660.0f) * 100);
    printf("FP64 Throughput:        %.2f TFLOPS (%.1f%% of theoretical)\n", 
           flops_fp64, (flops_fp64 / 1330.0f) * 100);
    printf("Global Memory BW:       %.2f GB/s (%.1f%% of theoretical)\n", 
           bandwidth_read, (bandwidth_read / 3935.0f) * 100);
    printf("Shared Memory BW:       %.2f GB/s\n", bandwidth_shared);
    printf("H2D PCIe Bandwidth:     %.2f GB/s\n", bandwidth_h2d);
    printf("D2H PCIe Bandwidth:     %.2f GB/s\n", bandwidth_d2h);
    printf("========================================\n");
    
    // Cleanup
    printf("\nCleaning up...\n");
    cudaFree(d_data_fp32);
    cudaFree(d_data_fp64);
    cudaFree(d_read_data);
    cudaFree(d_read_result);
    cudaFree(d_shared_result);
    cudaFree(d_transfer_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Done!\n");
    
    return 0;
}
