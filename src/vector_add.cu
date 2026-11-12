#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel function: Add two vectors
// Each thread adds one pair of elements
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    // Calculate the global index of this thread
    // This is the key CUDA indexing formula for 1D arrays
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check: ensure we don't go out of bounds
    if (idx < numElements) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * Host function to perform vector addition
 * 
 * GPU Memory Layout:
 * ┌─────────────────────┐
 * │  Vector A (GPU)     │  size: numElements * sizeof(float)
 * ├─────────────────────┤
 * │  Vector B (GPU)     │  size: numElements * sizeof(float)
 * ├─────────────────────┤
 * │  Vector C (GPU)     │  size: numElements * sizeof(float)
 * └─────────────────────┘
 * 
 * Execution Flow:
 * 1. Allocate GPU memory for A, B, C
 * 2. Copy A and B from CPU to GPU
 * 3. Launch kernel with multiple blocks and threads
 * 4. Each thread computes one element: C[i] = A[i] + B[i]
 * 5. Copy result C from GPU to CPU
 * 6. Free GPU memory
 */
int main(int argc, char *argv[]) {
    // Define number of elements
    int numElements = 50000;
    
    // Parse command line arguments
    if (argc > 1) {
        numElements = atoi(argv[1]);
    }
    
    // Calculate memory size
    size_t bytes = numElements * sizeof(float);
    
    printf("Vector Addition on GPU\n");
    printf("========================\n");
    printf("Number of elements: %d\n", numElements);
    printf("Memory required: %.2f MB\n", (float)bytes / (1024 * 1024));
    printf("\n");
    
    // Allocate host (CPU) memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize host vectors with random data
    printf("Initializing vectors...\n");
    for (int i = 0; i < numElements; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // Allocate device (GPU) memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    
    cudaError_t err = cudaMalloc((void **)&d_A, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for A: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&d_B, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for B: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&d_C, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    printf("GPU memory allocated successfully\n");
    
    // Copy data from host to device
    printf("Copying data from CPU to GPU...\n");
    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Calculate grid and block dimensions
    // Typical block size: 256 or 512 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("\nKernel Configuration:\n");
    printf("  Threads per block: %d\n", threadsPerBlock);
    printf("  Blocks per grid:   %d\n", blocksPerGrid);
    printf("  Total threads:     %d\n", threadsPerBlock * blocksPerGrid);
    
    // CUDA Event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start time
    cudaEventRecord(start, 0);
    
    // Launch the kernel
    printf("\nLaunching kernel...\n");
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // Record stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Calculate kernel execution time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Copy result from device to host
    printf("Copying results from GPU to CPU...\n");
    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Verify results
    printf("Verifying results...\n");
    bool correct = true;
    float tolerance = 1e-5f;
    
    for (int i = 0; i < numElements; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > tolerance) {
            fprintf(stderr, "Result verification failed at index %d!\n", i);
            fprintf(stderr, "  Expected: %f, Got: %f\n", expected, h_C[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("✓ All results verified successfully!\n");
    } else {
        printf("✗ Result verification failed!\n");
    }
    
    // Print sample results
    printf("\nSample results (first 10 elements):\n");
    printf("Index | A[i]     | B[i]     | C[i]=A[i]+B[i]\n");
    printf("------|----------|----------|---------------\n");
    for (int i = 0; i < (numElements < 10 ? numElements : 10); i++) {
        printf("%5d | %8.4f | %8.4f | %8.4f\n", i, h_A[i], h_B[i], h_C[i]);
    }
    
    // Performance metrics
    printf("\nPerformance Metrics:\n");
    float dataMovement = 3.0f * numElements * sizeof(float) / (1024 * 1024);  // 3 vectors
    float bandwidth = dataMovement / (milliseconds / 1000.0f);
    float flops = (float)numElements / (milliseconds / 1000.0f) / 1e9f;
    
    printf("  Data movement: %.2f MB\n", dataMovement);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth / 1024);
    printf("  Throughput: %.2f GFLOPS\n", flops);
    
    // Cleanup
    printf("\nCleaning up...\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Done!\n");
    
    return 0;
}
