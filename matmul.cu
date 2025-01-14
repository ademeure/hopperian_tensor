#include <cuda.h>
#include <sys/time.h>
//#include <iostream>
#include <stdio.h>
#include <cuda_bf16.h>
#include <assert.h>

#define ENABLE_CUBLAS
#define ENABLE_RANDOM
//#define ENABLE_TRUE_RANDOM
#define SLEEP_BETWEEN_KERNELS_SEC 0
#define REFERENCE_KERNEL 0
constexpr bool RUN_VERIF = true;
constexpr int max_size = 16384;
constexpr int prime = 3719;
int repeat_times = 10;

int get_time() {
  static int last_time = 0;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  int time = ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
  int diff = time - last_time;
  last_time = time;
  return diff;
}

typedef __nv_bfloat16 bf16;
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

#ifdef ENABLE_RANDOM
#include <random>
std::default_random_engine generator(69);
#endif

/*
#include "examples/matmul/matmul_1.cuh"
#include "examples/matmul/matmul_2.cuh"
#include "examples/matmul/matmul_3.cuh"
#include "examples/matmul/matmul_4.cuh"
#include "examples/matmul/matmul_5.cuh"
#include "examples/matmul/matmul_6.cuh"
#include "examples/matmul/matmul_7.cuh"
#include "examples/matmul/matmul_8.cuh"
#include "examples/matmul/matmul_9.cuh"
#include "examples/matmul/matmul_10.cuh"
#include "examples/matmul/matmul_11.cuh"
*/
#include "examples/matmul/matmul_10.cuh"

#ifdef ENABLE_CUBLAS
#include <cublas_v2.h>
cublasHandle_t cublas_handle;
void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  float alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16BF,
    N, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS error: %d\n", status);
    exit(1);
  }
}
#endif

void run_kernel(int kernel_num, int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB = nullptr) {
  switch (kernel_num) {
    case 0:
#ifdef ENABLE_CUBLAS
      runCublasGemmBF16(M, N, K, A, B, C);
#endif
      break;
    case 10:
      runKernel10(M, N, K, A, B, C, DB);
      break;

    /*
    case 1:
      runKernel1(M, N, K, A, B, C);
      break;
    case 2:
      runKernel2(M, N, K, A, B, C);
      break;
    case 3:
      runKernel3(M, N, K, A, B, C, DB);
      break;
    case 4:
      runKernel4(M, N, K, A, B, C, DB);
      break;
    case 5:
      runKernel5(M, N, K, A, B, C, DB);
      break;
    case 6:
      runKernel6(M, N, K, A, B, C, DB);
      break;
    case 7:
      runKernel7(M, N, K, A, B, C, DB);
      break;
    case 8:
      runKernel8(M, N, K, A, B, C, DB);
      break;
    case 9:
      runKernel9(M, N, K, A, B, C, DB);
      break;
    case 10:
      runKernel10(M, N, K, A, B, C, DB);
      break;
    case 11:
      runKernel11(M, N, K, A, B, C, DB);
      break;
    */
  }
}
void randomize_matrix(bf16 *mat, int N) {
#ifdef ENABLE_RANDOM
  std::normal_distribution<float> distribution(0, 1);
#ifdef ENABLE_TRUE_RANDOM
  for (int i = 0; i < N; i++) {
    mat[i] = distribution(generator);
  }
#else
  int i = 0;
  for (; i < prime; i++) {
    mat[i] = distribution(generator);
  }
  //printf("Pre-Memmove - time: %d\n", get_time());
  //memmove(mat+9479, mat, sizeof(bf16) * (N-9479));
  for (int multiplier = 1; i < N-(prime * multiplier); i += prime * multiplier, multiplier *= 2) {
    memcpy(mat+i, mat, sizeof(bf16) * prime);
  }
  for (; i < N-prime; i += prime) {
    memcpy(mat+i, mat, sizeof(bf16) * prime);
  }
  for (; i < N; i++) {
    mat[i] = mat[i-prime];
  }
  //printf("Post-Duplication - time: %d\n", get_time());
#endif
#else
  cudaMemset(mat, 0, sizeof(bf16) * N);
#endif
}

/*
bool verify_matrix(bf16 *matRef, bf16 *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    int r = i / 8192, c = i % 8192;
    int it = c*8192+r;
    diff = std::fabs(__bfloat162float(matRef[i] - matOut[i]));
    if (diff > 0.1) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
      __bfloat162float(matRef[i]), __bfloat162float(matOut[i]), diff, i);
      return false;
    }
  }
  return true;
}
*/

__global__ void verify_matrix_kernel(bf16 *matRef, bf16 *matOut, int *result, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float diff = (float)matRef[i] - (float)matOut[i];
    if (diff > 0.1) {
      *result = 0;
    }
  }
}

__global__ void warmupKernel() {
  __shared__ int s[100];
  s[0] += s[1];
}

int main() {
  get_time();
  //warmupKernel<<<32, 32>>>();
  //printf("Warmed up - time: %d\n", get_time());
  long m = max_size, n = max_size, k = max_size;

  bf16 *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;  // host matriceRas
  bf16 *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; // device matrices

  int *DB = nullptr; int *dDB = nullptr;

  A = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  B = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C_ref = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  DB = (int *)malloc(sizeof(int) * max_size * 128);

  //printf("Randomizing matrices - time: %d\n", get_time());
  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);
  //printf("Randomized matrices - time: %d\n", get_time());

  int* result;
  int result_host;
  cudaMalloc((void**)&result, sizeof(int));
  cudaCheck(cudaMalloc((void **)&dDB, sizeof(int) * max_size * 128));
  cudaCheck(cudaMalloc((void **)&dA, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(bf16) * max_size * max_size));

  //printf("Post-Malloc - time: %d\n", get_time());

  cudaCheck(cudaMemcpyAsync(dA, A, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dB, B, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
  //printf("Post-Memcpy - time: %d\n", get_time());

#ifdef ENABLE_CUBLAS
  cublasCreate(&cublas_handle);
  //printf("Cublas created - time: %d\n", get_time());
#endif

  timespec ts_second;
  ts_second.tv_sec = SLEEP_BETWEEN_KERNELS_SEC;
  ts_second.tv_nsec = 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time;

  bool first_run = true;
  bool run_verif = RUN_VERIF;
  //printf("Verifying kernels - time: %d\n", get_time());
  for (int kernel_num : {10}) {
    printf("\nKERNEL %d\n", kernel_num);

    // Give the GPU some rest to avoid thermal throttling
    if (!first_run) {
      nanosleep(&ts_second, NULL);
    }
    first_run = false;

    // Verify against cuBLAS. Also serves as a warmup step
    if (run_verif) {
      cudaCheck(cudaMemset(dC, 0, sizeof(bf16) * max_size * max_size));
      cudaCheck(cudaMemset(dC_ref, 0, sizeof(bf16) * max_size * max_size));
      cudaCheck(cudaMemset(dDB, ~0, sizeof(int) * max_size * 128));
      cudaCheck(cudaMemset(result, 1, sizeof(int)));

      run_kernel(kernel_num, m, n, k, dA, dB, dC);
      run_kernel(REFERENCE_KERNEL, m, n, k, dA, dB, dC_ref);

      verify_matrix_kernel<<<CEIL_DIV(m * n, 1024), 1024>>>(dC_ref, dC, result, m * n);
      cudaMemcpy(&result_host, result, sizeof(int), cudaMemcpyDeviceToHost); // can only be async because next memcpy isn't
      cudaMemcpy(DB, dDB, sizeof(int) * max_size * 8, cudaMemcpyDeviceToHost);
      printf("\n=======> Kernel %d -> VERIFICATION: %s\n\n", kernel_num, result_host ? "TRUE" : "FALSE");

      /*
      int i = 0;
      long sumLoad = 0, cntLoad = 0;
      long sumCompute = 0, cntCompute = 0;
      long sumStore = 0, cntStore = 0;
      int times = 0;
      while (DB[i] != ~0) {
        sumLoad += DB[i], cntLoad += DB[i + 1];
        sumCompute += DB[i + 2], cntCompute += DB[i + 3];
        sumStore += DB[i + 4], cntStore += DB[i + 5];
        i += 6;
        times++;
      }
      if (times > 0) {
        printf("Load: %f, Compute: %f,  Store: %f, Datapoints: %d\n", (sumLoad + .0) / cntLoad, (sumCompute + .0) / cntCompute, (sumStore + .0) / cntStore, times);
      }
      */
    }

    printf("Benchmarking kernel %d - time: %d\n", kernel_num, get_time());

    // Benchmark
    cudaEventRecord(start);
    for (int j = 0; j < repeat_times; j++) {
      run_kernel(kernel_num, m, n, k, dA, dB, dC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    long flops = (2LL * m) * (n * k);
    printf(
        "=======> Average elapsed time: (%7.6f) s, performance: (%7.1f) TFLOPS. size: (%ld).\n\n",
        elapsed_time / 1000.0 / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);

    printf("Benchmarked kernel %d - time: %d\n", kernel_num, get_time());
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cudaFree(result);
  return 0;
};