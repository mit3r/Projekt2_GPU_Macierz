#include <assert.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <helpers/helper_cuda.h>
#include <helpers/helper_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DISPATCH_KERNEL(BS_VAL, N_VAL)                                \
  case N_VAL:                                                         \
    MatrixMulCUDA_NCols<BS_VAL, N_VAL><<<grid, threads, 0, stream>>>( \
        d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.x);                    \
    break;

template <int BLOCK_SIZE, int N>
__global__ void MatrixMulCUDA_NCols(float* C, float* A, float* B, int wA,
                                    int hA, int wB) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Każdy wątek odpowiada za macierz N x N wyników
  float Csub[N][N] = {0.0f};

  for (int k = 0; k < wA; k += BLOCK_SIZE) {
    // Pamięć współdzielona dostosowana do wczytywania 2D bloków
    __shared__ float As[BLOCK_SIZE * N][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE * N];

    // Wczytywanie macierzy A
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int row = by * BLOCK_SIZE * N + i * BLOCK_SIZE + ty;
      int col = k + tx;

      As[i * BLOCK_SIZE + ty][tx] = A[row * wA + col];
    }

// Wczytywanie macierzy B
#pragma unroll
    for (int j = 0; j < N; ++j) {
      int row = k + ty;
      int col = bx * BLOCK_SIZE * N + j * BLOCK_SIZE + tx;

      Bs[ty][j * BLOCK_SIZE + tx] = B[row * wB + col];
    }

    __syncthreads();

    // Wyliczenie submacierzy wyników Csub
#pragma unroll
    for (int d = 0; d < BLOCK_SIZE; ++d) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        float a_val = As[i * BLOCK_SIZE + ty][d];
#pragma unroll
        for (int j = 0; j < N; ++j) {
          Csub[i][j] += a_val * Bs[d][j * BLOCK_SIZE + tx];
        }
      }
    }

    __syncthreads();
  }

  // Zapisywanie wyliczonych wartości C na odpowiednie miejsce w pamięci
  // globalnej
#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      int row = by * BLOCK_SIZE * N + i * BLOCK_SIZE + ty;
      int col = bx * BLOCK_SIZE * N + j * BLOCK_SIZE + tx;
      if (row < hA && col < wB) {
        C[row * wB + col] = Csub[i][j];
      }
    }
  }
}

void ConstantInit(float* data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

void RandomInitRange(float* data, int size, float min_val, float max_val) {
  for (int i = 0; i < size; ++i) {
    float normalized =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    data[i] = min_val + normalized * (max_val - min_val);
  }
}

void IdentityInit(float* data, int width, int height) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      data[row * width + col] = (row == col) ? 1.0f : 0.0f;
    }
  }
}

template <int BLOCK_SIZE>
int MatrixMultiply(int n, const dim3& dimsA, const dim3& dimsB,
                   const float* h_A, const float* h_B, float* h_C_out) {
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;

  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;

  unsigned int size_C = dimsB.x * dimsA.y;
  unsigned int mem_size_C = sizeof(float) * size_C;

  cudaStream_t stream;

  float *d_A, *d_B, *d_C;

  dim3 dimsC(dimsB.x, dimsA.y, 1);
  mem_size_C = dimsC.x * dimsC.y * sizeof(float);

  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  // Obliczenie gridu ze wsparciem dla przypadków niepodzielnych brzegów
  dim3 grid((dimsB.x) / (threads.x * n),
            (dimsA.y) / (threads.y * n));

  printf("Computing result using CUDA Kernel...\n");

  // Performs warmup operation using matrixMul CUDA kernel
  switch (n) {
    DISPATCH_KERNEL(BLOCK_SIZE, 1);
    DISPATCH_KERNEL(BLOCK_SIZE, 2);
    DISPATCH_KERNEL(BLOCK_SIZE, 3);
    DISPATCH_KERNEL(BLOCK_SIZE, 4);
    DISPATCH_KERNEL(BLOCK_SIZE, 5);
    DISPATCH_KERNEL(BLOCK_SIZE, 6);
    // DISPATCH_KERNEL(BLOCK_SIZE, 8); // shared memory limit
    // DISPATCH_KERNEL(BLOCK_SIZE, 16);
    // DISPATCH_KERNEL(BLOCK_SIZE, 32);
    // DISPATCH_KERNEL(BLOCK_SIZE, 64);
  default:
    fprintf(stderr, "Error: n is not included\n");
    return EXIT_FAILURE;
  }

  printf("done\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  int nIter = 1;
  for (int j = 0; j < nIter; j++) {
    switch (n) {
      DISPATCH_KERNEL(BLOCK_SIZE, 1);
      DISPATCH_KERNEL(BLOCK_SIZE, 2);
      DISPATCH_KERNEL(BLOCK_SIZE, 3);
      DISPATCH_KERNEL(BLOCK_SIZE, 4);
      DISPATCH_KERNEL(BLOCK_SIZE, 5);
      DISPATCH_KERNEL(BLOCK_SIZE, 6);
      // DISPATCH_KERNEL(BLOCK_SIZE, 8); // shared memory limit
      // DISPATCH_KERNEL(BLOCK_SIZE, 16);
      // DISPATCH_KERNEL(BLOCK_SIZE, 32);
      // DISPATCH_KERNEL(BLOCK_SIZE, 64);
    default:
      fprintf(stderr, "Error: n is not included\n");
      return EXIT_FAILURE;
    }
  }

  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

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

  checkCudaErrors(cudaMemcpyAsync(h_C_out, d_C, mem_size_C,
                                  cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Clean up memory
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return EXIT_SUCCESS;
}

/**
 * Program main
 */
int main(int argc, char** argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
      checkCmdLineFlag(argc, (const char**)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -n=N (N is the 2D block multiplier, 1-6)\n");
    printf(
        "  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  int dev = findCudaDevice(argc, (const char**)argv);

  int n = 1;
  constexpr int block_size = 32;

  // Nowe domyślne wielkości macierzy: 3072 x 3072
  dim3 dimsA(3072, 3072, 1);
  dim3 dimsB(3072, 3072, 1);

  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int size_C = dimsB.x * dimsA.y;

  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int mem_size_B = sizeof(float) * size_B;
  unsigned int mem_size_C = sizeof(float) * size_C;

  float* h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  float* h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  float* h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
  srand(static_cast<unsigned int>(time(NULL)));

  // work size
  if (checkCmdLineFlag(argc, (const char**)argv, "n")) {
    n = getCmdLineArgumentInt(argc, (const char**)argv, "n");
  }

  checkCudaErrors(cudaProfilerStart());
  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
         dimsB.y);

  printf("Test 1: synthetic A=1, B=0.01\n");
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, 0.01f);
  int matrix_result =
      MatrixMultiply<block_size>(n, dimsA, dimsB, h_A, h_B, h_C);
  bool test1_ok = true;
  if (matrix_result == EXIT_SUCCESS) {
    const float valB = 0.01f;
    double eps = 1.e-4;
    for (int i = 0; i < static_cast<int>(size_C); ++i) {
      double abs_err = fabs(h_C[i] - (dimsA.x * valB));
      double dot_length = dimsA.x;
      double abs_val = fabs(h_C[i]);
      double rel_err = abs_err / abs_val / dot_length;
      if (rel_err > eps) {
        test1_ok = false;
        printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
               h_C[i], dimsA.x * valB, eps);
        break;
      }
    }
  } else {
    test1_ok = false;
  }
  printf("Test 1 result: %s\n", test1_ok ? "PASS" : "FAIL");

  printf("Test 2: random A in [1,5], B=I\n");
  RandomInitRange(h_A, size_A, 1.0f, 5.0f);
  IdentityInit(h_B, dimsB.x, dimsB.y);
  int matrix_result_second =
      MatrixMultiply<block_size>(n, dimsA, dimsB, h_A, h_B, h_C);
  bool test2_ok = true;
  if (matrix_result_second == EXIT_SUCCESS) {
    double eps = 1.e-4;
    for (int i = 0; i < static_cast<int>(size_C); ++i) {
      double abs_err = fabs(h_C[i] - h_A[i]);
      double abs_val = fabs(h_C[i]);
      double ref_val = fabs(h_A[i]);
      double denom = abs_val > ref_val ? abs_val : ref_val;
      if (denom == 0.0)
        denom = 1.0;
      double rel_err = abs_err / denom;
      if (rel_err > eps) {
        test2_ok = false;
        printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
               h_C[i], h_A[i], eps);
        break;
      }
    }
  } else {
    test2_ok = false;
  }
  printf("Test 2 result: %s\n", test2_ok ? "PASS" : "FAIL");

  checkCudaErrors(cudaProfilerStop());
  printf("Profiler stopped.\n");

  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));

  if (test1_ok && test2_ok) {
    exit(EXIT_SUCCESS);
  }

  exit(EXIT_FAILURE);
}