#include <assert.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <helpers/helper_cuda.h>
#include <helpers/helper_functions.h>
#include <stdio.h>

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE, int N>
__global__ void MatrixMulCUDA_NCols(float* C, float* A,
                                    float* B, int wA,
                                    int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A in row processed by the block
  int aBegin = wA * BLOCK_SIZE * by * N;

  // Index of the last sub-matrix of A in row processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B in column processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    Csub[i] = 0.0f;
  }

  // Loop over all the sub-matrices of A and B
  // required to compute partial sums of the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {
    // [warp * N][thread]
    __shared__ float As[BLOCK_SIZE * N][BLOCK_SIZE];

    // [warp][thread]
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
#pragma unroll
    for (int i = 0; i < N; ++i) {
      As[ty + i * BLOCK_SIZE][tx] = A[a + wA * (ty + i * BLOCK_SIZE) + tx];
    }

    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      float b_val = Bs[k][tx];

#pragma unroll
      for (int i = 0; i < N; ++i) {
        Csub[i] += As[ty + i * BLOCK_SIZE][k] * b_val;
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int cBegin = wB * BLOCK_SIZE * by * N + BLOCK_SIZE * bx;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    C[cBegin + wB * BLOCK_SIZE * i + wB * ty + tx] = Csub[i];
  }
}

void ConstantInit(float* data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char** argv,
                   int block_size, const dim3& dimsA,
                   const dim3& dimsB) {
  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  cudaStream_t stream;

  // Initialize host memory
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);

  // Allocate device memory
  float *d_A, *d_B, *d_C;

  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float* h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C));
  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // copy host memory to device
  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

  // Setup execution parameters
  const int N = 2;
  dim3 threads(block_size, block_size);
  if ((dimsA.y % (threads.y * N)) != 0 || (dimsB.x % threads.x) != 0) {
    fprintf(stderr,
            "Error: dimensions must satisfy hA %% (bs*N) == 0 and wB %% bs == 0. "
            "(hA=%u, wB=%u, bs=%u, N=%d)\n",
            dimsA.y, dimsB.x, threads.x, N);
    return EXIT_FAILURE;
  }
  // One block computes N row-tiles from A and one column-tile from B.
  dim3 grid(dimsB.x / threads.x, dimsA.y / (threads.y * N));

  // Create and start timer
  printf("Computing result using CUDA Kernel...\n");

  // Performs warmup operation using matrixMul CUDA kernel
  if (block_size == 8) {
    MatrixMulCUDA_NCols<8, N>
        <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  } else if (block_size == 16) {
    MatrixMulCUDA_NCols<16, N>
        <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  } else if (block_size == 32) {
    MatrixMulCUDA_NCols<32, N>
        <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  }

  printf("done\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  // Execute the kernel
  int nIter = 1;

  for (int j = 0; j < nIter; j++) {
    if (block_size == 8) {
      MatrixMulCUDA_NCols<8, N>
          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    } else if (block_size == 16) {
      MatrixMulCUDA_NCols<16, N>
          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    } else if (block_size == 32) {
      MatrixMulCUDA_NCols<32, N>
          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Checking computed result for correctness: ");
  bool correct = true;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-4;  // 1.e-6;  // machine zero

  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
             i, h_C[i], dimsA.x * valB, eps);
      correct = false;
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // Clean up memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  printf(
      "\nNOTE: The CUDA Samples are not meant for performance "
      "measurements. Results may vary when GPU Boost is enabled.\n");

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}

/**
 * Program main
 */
int main(int argc, char** argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
      checkCmdLineFlag(argc, (const char**)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf(
        "  Note: Outer matrix dimensions of A & B matrices"
        " must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  int dev = findCudaDevice(argc, (const char**)argv);

  int block_size = 32;

  dim3 dimsA(100 * block_size, 100 * block_size, 1);
  dim3 dimsB(100 * block_size, 100 * block_size, 1);

  // width of Matrix A
  if (checkCmdLineFlag(argc, (const char**)argv, "wA")) {
    dimsA.x = getCmdLineArgumentInt(argc, (const char**)argv, "wA");
  }

  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char**)argv, "hA")) {
    dimsA.y = getCmdLineArgumentInt(argc, (const char**)argv, "hA");
  }

  // width of Matrix B
  if (checkCmdLineFlag(argc, (const char**)argv, "wB")) {
    dimsB.x = getCmdLineArgumentInt(argc, (const char**)argv, "wB");
  }

  // height of Matrix B
  if (checkCmdLineFlag(argc, (const char**)argv, "hB")) {
    dimsB.y = getCmdLineArgumentInt(argc, (const char**)argv, "hB");
  }

  // block size
  if (checkCmdLineFlag(argc, (const char**)argv, "bs")) {
    block_size = getCmdLineArgumentInt(argc, (const char**)argv, "bs");
  }

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  if (block_size != 8 && block_size != 16 && block_size != 32) {
    printf("Error: block size must be either 8, 16 or 32. (%d)\n", block_size);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}