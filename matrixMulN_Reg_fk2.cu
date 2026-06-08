#include <assert.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <helpers/helper_cuda.h>
#include <helpers/helper_functions.h>
#include <stdio.h>

static bool LoadBinaryMatrix(const char* path, float* data, size_t count) {
  FILE* fp = NULL;
  FOPEN(fp, path, "rb");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open input file: %s\n", path);
    return false;
  }

  size_t read_count = fread(data, sizeof(float), count, fp);
  fclose(fp);

  if (read_count != count) {
    fprintf(stderr,
            "Failed to read %zu floats from input file: %s (read %zu)\n",
            count, path, read_count);
    return false;
  }

  return true;
}

#define DISPATCH_KERNEL(N_VAL)                                                                                \
  case N_VAL:                                                                                                 \
    switch (block_size) {                                                                                     \
    case 8:                                                                                                   \
      MatrixMulCUDA_NCols<8, N_VAL><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.x);  \
      break;                                                                                                  \
    case 16:                                                                                                  \
      MatrixMulCUDA_NCols<16, N_VAL><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.x); \
      break;                                                                                                  \
    case 32:                                                                                                  \
      MatrixMulCUDA_NCols<32, N_VAL><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.x); \
      break;                                                                                                  \
    }                                                                                                         \
    break;

template <int BLOCK_SIZE, int N>
__global__ void MatrixMulCUDA_NCols(float* C, float* A, float* B, int wA, int hA, int wB) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Każdy wątek odpowiada za macierz N x N wyników
  float Csub[N][N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      Csub[i][j] = 0.0f;
    }
  }

  for (int k = 0; k < wA; k += BLOCK_SIZE) {
    // Pamięć współdzielona dostosowana do wczytywania 2D bloków
    __shared__ float As[BLOCK_SIZE * N][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE * N];

    // Wczytywanie macierzy A (Zabezpieczenie przed wyjściem poza granice)
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int row = by * BLOCK_SIZE * N + i * BLOCK_SIZE + ty;
      int col = k + tx;
      if (row < hA && col < wA) {
        As[i * BLOCK_SIZE + ty][tx] = A[row * wA + col];
      } else {
        As[i * BLOCK_SIZE + ty][tx] = 0.0f;
      }
    }

    // Wczytywanie macierzy B (Zabezpieczenie przed wyjściem poza granice)
#pragma unroll
    for (int j = 0; j < N; ++j) {
      int row = k + ty;
      int col = bx * BLOCK_SIZE * N + j * BLOCK_SIZE + tx;
      if (row < wA && col < wB) {
        Bs[ty][j * BLOCK_SIZE + tx] = B[row * wB + col];
      } else {
        Bs[ty][j * BLOCK_SIZE + tx] = 0.0f;
      }
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

  // Zapisywanie wyliczonych wartości C na odpowiednie miejsce w pamięci globalnej
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

int MatrixMultiply(int block_size, int n, const dim3& dimsA, const dim3& dimsB,
                   const float* h_A, const float* h_B,
                   const float* h_C_ref, bool hasReferenceFile) {
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
  float* h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

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

  dim3 threads(block_size, block_size);

  // Obliczenie gridu ze wsparciem dla przypadków niepodzielnych brzegów
  dim3 grid((dimsB.x + threads.x * n - 1) / (threads.x * n),
            (dimsA.y + threads.y * n - 1) / (threads.y * n));

  printf("Computing result using CUDA Kernel...\n");

  // Performs warmup operation using matrixMul CUDA kernel
  switch (n) {
    DISPATCH_KERNEL(1);
    DISPATCH_KERNEL(2);
    DISPATCH_KERNEL(3);
    DISPATCH_KERNEL(4);
    DISPATCH_KERNEL(5);
    DISPATCH_KERNEL(6);
    DISPATCH_KERNEL(7);
    DISPATCH_KERNEL(8);
  default:
    fprintf(stderr, "Error: n must be between 1 and 8. (%d)\n", n);
    return EXIT_FAILURE;
  }

  printf("done\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  int nIter = 1;
  for (int j = 0; j < nIter; j++) {
    switch (n) {
      DISPATCH_KERNEL(1);
      DISPATCH_KERNEL(2);
      DISPATCH_KERNEL(3);
      DISPATCH_KERNEL(4);
      DISPATCH_KERNEL(5);
      DISPATCH_KERNEL(6);
      DISPATCH_KERNEL(7);
      DISPATCH_KERNEL(8);
    default:
      fprintf(stderr, "Error: n must be between 1 and 8. (%d)\n", n);
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

  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Checking computed result for correctness: ");
  bool correct = true;
  if (hasReferenceFile) {
    double eps = 1.e-4;

    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
      double abs_err = fabs(h_C[i] - h_C_ref[i]);
      double abs_val = fabs(h_C[i]);
      double ref_val = fabs(h_C_ref[i]);
      double denom = abs_val > ref_val ? abs_val : ref_val;
      if (denom == 0.0) denom = 1.0;
      double rel_err = abs_err / denom;

      if (rel_err > eps) {
        printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
               i, h_C[i], h_C_ref[i], eps);
        correct = false;
      }
    }
  } else {
    // Default behavior: compare to synthetic input case used previously
    const float valB = 0.01f;
    double eps = 1.e-4;  // same tolerance as other samples

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
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // Clean up memory
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
    printf("      -bs=BlockSize (Block size is 8, 16 or 32)\n");
    printf("      -n=N (N is the 2D block multiplier, 1-8)\n");
    printf("      -ina=path -inb=path (binary float32 inputs, 4096x4096 each)\n");
    printf("      -cin=path (binary float32 reference C, 4096x4096)\n");
    printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  int dev = findCudaDevice(argc, (const char**)argv);

  int block_size = 32, n = 2;

  // Nowe domyślne wielkości macierzy: 4096 x 4096
  dim3 dimsA(4096, 4096, 1);
  dim3 dimsB(4096, 4096, 1);

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
  float* h_C_ref = NULL;
  bool hasReferenceFile = false;
  char* inAPath = NULL;
  char* inBPath = NULL;
  char* inCPath = NULL;

  // block size
  if (checkCmdLineFlag(argc, (const char**)argv, "bs")) {
    block_size = getCmdLineArgumentInt(argc, (const char**)argv, "bs");
  }

  // work size
  if (checkCmdLineFlag(argc, (const char**)argv, "n")) {
    n = getCmdLineArgumentInt(argc, (const char**)argv, "n");
  }

  if (block_size != 8 && block_size != 16 && block_size != 32) {
    printf("Error: block size must be either 8, 16 or 32. (%d)\n", block_size);
    exit(EXIT_FAILURE);
  }

  if (n < 1 || n > 8) {
    printf("Error: n must be between 1 and 8. (%d)\n", n);
    exit(EXIT_FAILURE);
  }

  if (getCmdLineArgumentString(argc, (const char**)argv, "ina", &inAPath) ||
      getCmdLineArgumentString(argc, (const char**)argv, "inb", &inBPath)) {
    if (inAPath == NULL || inBPath == NULL) {
      printf("Error: both -ina and -inb must be provided together.\n");
      exit(EXIT_FAILURE);
    }
    if (!LoadBinaryMatrix(inAPath, h_A, size_A) ||
        !LoadBinaryMatrix(inBPath, h_B, size_B)) {
      exit(EXIT_FAILURE);
    }
  } else {
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);
  }

  if (getCmdLineArgumentString(argc, (const char**)argv, "cin", &inCPath)) {
    hasReferenceFile = true;
    checkCudaErrors(cudaMallocHost(&h_C_ref, mem_size_C));
    if (!LoadBinaryMatrix(inCPath, h_C_ref, size_C)) {
      exit(EXIT_FAILURE);
    }
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(
      block_size,
      n, dimsA, dimsB, h_A, h_B,
      h_C_ref, hasReferenceFile);
  checkCudaErrors(cudaProfilerStop());

  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  if (h_C_ref != NULL) {
    checkCudaErrors(cudaFreeHost(h_C_ref));
  }

  exit(matrix_result);
}