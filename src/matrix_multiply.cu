#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * Matrix Multiplication using CUDA
 * 
 * This example demonstrates:
 * 1. 2D block and grid configuration
 * 2. 2D index calculation
 * 3. Shared memory for optimization
 * 4. Using cuBLAS library for comparison
 * 
 * Matrix dimensions:
 * C[M x N] = A[M x K] * B[K x N]
 */

// Naive kernel: Each thread computes one element of C
__global__ void matrixMultiplyNaive(float *A, float *B, float *C, 
                                     int M, int K, int N) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row from A and column from B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

/**
 * Optimized kernel using shared memory
 * Reduces global memory access significantly
 * 
 * Block size: 16x16 (256 threads)
 * Tile size: 16x16
 */
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                      int M, int K, int N) {
    // Shared memory tiles
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];
    
    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    
    float sum = 0.0f;
    
    // Process tiles of size 16x16
    for (int tileIdx = 0; tileIdx < (K + 15) / 16; tileIdx++) {
        // Load tile from A into shared memory
        if (row < M && tileIdx * 16 + localCol < K) {
            tileA[localRow][localCol] = A[row * K + tileIdx * 16 + localCol];
        } else {
            tileA[localRow][localCol] = 0.0f;
        }
        
        // Load tile from B into shared memory
        if (tileIdx * 16 + localRow < K && col < N) {
            tileB[localRow][localCol] = B[(tileIdx * 16 + localRow) * N + col];
        } else {
            tileB[localRow][localCol] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded data
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < 16; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }
        
        // Synchronize before next iteration
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Utility function to initialize matrix
void initMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Utility function to verify results
bool verifyResults(float *C, float *C_ref, int M, int N) {
    float tolerance = 1e-3f;
    
    for (int i = 0; i < M * N; i++) {
        if (fabs(C[i] - C_ref[i]) > tolerance) {
            printf("Mismatch at index %d: got %f, expected %f\n", i, C[i], C_ref[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    // Matrix dimensions: C[M x N] = A[M x K] * B[K x N]
    int M = 1024;  // Height of A and C
    int K = 1024;  // Width of A, height of B
    int N = 1024;  // Width of B and C
    
    // Parse command line arguments
    if (argc > 3) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }
    
    printf("Matrix Multiplication on GPU (CUDA)\n");
    printf("====================================\n");
    printf("Matrix dimensions: C[%d x %d] = A[%d x %d] * B[%d x %d]\n", M, N, M, K, K, N);
    printf("\n");
    
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    
    printf("Memory allocation:\n");
    printf("  Matrix A: %.2f MB\n", bytesA / (1024 * 1024.0f));
    printf("  Matrix B: %.2f MB\n", bytesB / (1024 * 1024.0f));
    printf("  Matrix C: %.2f MB\n", bytesC / (1024 * 1024.0f));
    printf("  Total:    %.2f MB\n", (bytesA + bytesB + bytesC) / (1024 * 1024.0f));
    printf("\n");
    
    // Allocate host memory
    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);
    float *h_C_ref = (float *)malloc(bytesC);
    
    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);
    
    // Copy data to device
    printf("Copying data to GPU...\n");
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
    
    // ============================================================
    // Method 1: Naive CUDA kernel
    // ============================================================
    printf("\n--- Method 1: Naive CUDA Kernel ---\n");
    
    dim3 blockDim(16, 16);  // 16x16 = 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    
    printf("Block configuration: %dx%d\n", blockDim.x, blockDim.y);
    printf("Grid configuration: %dx%d\n", gridDim.x, gridDim.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrixMultiplyNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float msNaive = 0.0f;
    cudaEventElapsedTime(&msNaive, start, stop);
    printf("Naive kernel time: %.3f ms\n", msNaive);
    
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    
    // ============================================================
    // Method 2: Optimized CUDA kernel with shared memory
    // ============================================================
    printf("\n--- Method 2: Optimized CUDA Kernel (Shared Memory) ---\n");
    
    cudaEventRecord(start);
    matrixMultiplyShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float msShared = 0.0f;
    cudaEventElapsedTime(&msShared, start, stop);
    printf("Shared memory kernel time: %.3f ms\n", msShared);
    
    float *h_C_shared = (float *)malloc(bytesC);
    cudaMemcpy(h_C_shared, d_C, bytesC, cudaMemcpyDeviceToHost);
    
    // ============================================================
    // Method 3: cuBLAS library (for reference)
    // ============================================================
    printf("\n--- Method 3: cuBLAS Library ---\n");
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float msCublas = 0.0f;
    cudaEventElapsedTime(&msCublas, start, stop);
    printf("cuBLAS kernel time: %.3f ms\n", msCublas);
    
    cudaMemcpy(h_C_ref, d_C, bytesC, cudaMemcpyDeviceToHost);
    
    // ============================================================
    // Verify results
    // ============================================================
    printf("\n--- Results Verification ---\n");
    
    bool valid_shared = verifyResults(h_C_shared, h_C_ref, M, N);
    printf("Shared memory kernel: %s\n", valid_shared ? "PASSED" : "FAILED");
    
    bool valid_naive = verifyResults(h_C, h_C_ref, M, N);
    printf("Naive kernel: %s\n", valid_naive ? "PASSED" : "FAILED");
    
    // ============================================================
    // Performance Analysis
    // ============================================================
    printf("\n--- Performance Summary ---\n");
    
    float flops = 2.0f * M * K * N;  // 2 ops per multiplication
    float gflops_naive = flops / (msNaive * 1e6f);
    float gflops_shared = flops / (msShared * 1e6f);
    float gflops_cublas = flops / (msCublas * 1e6f);
    
    printf("Naive kernel:         %.2f GFLOPS (%.3f ms)\n", gflops_naive, msNaive);
    printf("Shared memory kernel: %.2f GFLOPS (%.3f ms) - %.2fx faster\n", 
           gflops_shared, msShared, msNaive / msShared);
    printf("cuBLAS kernel:        %.2f GFLOPS (%.3f ms) - %.2fx faster\n", 
           gflops_cublas, msCublas, msNaive / msCublas);
    
    // ============================================================
    // Cleanup
    // ============================================================
    printf("\nCleaning up...\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    free(h_C_shared);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    
    printf("Done!\n");
    
    return 0;
}
