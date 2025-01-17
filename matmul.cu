#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <assert.h>

#define ENABLE_CUBLAS
#define ENABLE_RANDOM
//#define ENABLE_TRUE_RANDOM
#define SLEEP_BETWEEN_KERNELS_SEC 1 // optional rest to avoid thermal throttling between kernels
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

// ...
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
  for (int multiplier = 1; i < N-(prime * multiplier); i += prime * multiplier, multiplier *= 2) {
    memcpy(mat+i, mat, sizeof(bf16) * prime);
  }
  for (; i < N-prime; i += prime) {
    memcpy(mat+i, mat, sizeof(bf16) * prime);
  }
  for (; i < N; i++) {
    mat[i] = mat[i-prime];
  }
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
    float diff = fabs((float)matRef[i] - (float)matOut[i]);
    if (diff > 0.1) {
      // accept result if it looks like RELU
      if ((float)matRef[i] > 0.0f || (float)matOut[i] != 0.0f) {
        *result = 0;
      }
    }
  }
}

int main() {
  get_time();
  long m = max_size, n = max_size, k = max_size;

  bf16 *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;  // host matrices
  bf16 *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; // device matrices

  int *DB = nullptr; int *dDB = nullptr;

  A = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  B = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C_ref = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  DB = (int *)malloc(sizeof(int) * max_size * 128);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  int* result;
  int result_host;
  cudaMalloc((void**)&result, sizeof(int));
  cudaCheck(cudaMalloc((void **)&dDB, sizeof(int) * max_size * 128));
  cudaCheck(cudaMalloc((void **)&dA, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(bf16) * max_size * max_size));

  cudaCheck(cudaMemcpyAsync(dA, A, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dB, B, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));

#ifdef ENABLE_CUBLAS
  cublasCreate(&cublas_handle);
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
  for (int kernel_num : {10,10,10}) {
    printf("\nKERNEL %d\n", kernel_num);

    if (!first_run) {
      nanosleep(&ts_second, NULL); // optional rest to avoid thermal throttling between kernels
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
      printf("\n=======> Kernel %d -> VERIFICATION: %s\n\n", kernel_num, result_host ? "OK" : "!!!!! FAILED !!!!!");
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
    printf( "=======> Average elapsed time: (%7.6f) s, performance: (%7.1f) TFLOPS. size: (%ld).\n\n",
        elapsed_time / 1000.0 / repeat_times, (repeat_times * flops * 1e-9) / elapsed_time, m);
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