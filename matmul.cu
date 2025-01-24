#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>
#include <cuda_bf16.h>
//#include <cuda_fp8.h>
#include <assert.h>

typedef __nv_bfloat16 floatX;
typedef __nv_bfloat16 floatP;
#define CUBLAS_FLOATX CUDA_R_16BF

constexpr bool ENABLE_C_INPUT = false;
constexpr float ENABLE_ABSMAX_SCALING = 0.0f;

#define ENABLE_CUBLAS
#define ENABLE_RANDOM
#define ENABLE_TRUE_RANDOM
#define SLEEP_BETWEEN_KERNELS_SEC 1 // optional rest to avoid thermal throttling between kernels
#define REFERENCE_KERNEL 0
constexpr bool RUN_VERIF = true;
constexpr int max_size = 16384;
constexpr int prime = 3719;
int repeat_times = 128;

int get_time() {
  static int last_time = 0;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  int time = ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
  int diff = time - last_time;
  last_time = time;
  return diff;
}

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
void runCublasGemmBF16(int M, int N, int K, floatX *A, floatX *B, floatX *C) {
  float alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUBLAS_FLOATX,
    N, B, CUBLAS_FLOATX, K, &beta, C, CUBLAS_FLOATX, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS error: %d\n", status);
    exit(1);
  }
}
#endif

void run_kernel(int kernel_num, int M, int N, int K, floatX *A, floatX *B, floatX *C, floatX *I, unsigned int* scalar_gpu) {
  switch (kernel_num) {
    case 0:
#ifdef ENABLE_CUBLAS
      runCublasGemmBF16(M, N, K, A, B, C);
#endif
      break;
    case 10:
      runKernel10(M, N, K, A, B, C, I, scalar_gpu);
      break;
  }
}

void randomize_matrix(floatX *mat, int N, float scale=1.0f) {
  if (scale == 0.0f) {
    for (int i = 0; i < N; i++) {
      mat[i] = (floatX)(i*1000);
    }
    return;
  }
#ifdef ENABLE_RANDOM
  std::normal_distribution<float> distribution(0, scale);
#ifdef ENABLE_TRUE_RANDOM
  for (int i = 0; i < N; i++) {
    mat[i] = (floatX)(distribution(generator) + 0.01f)  ;
  }
#else
  int i = 0;
  for (; i < prime; i++) {
    mat[i] = distribution(generator);
  }
  for (int multiplier = 1; i < N-(prime * multiplier); i += prime * multiplier, multiplier *= 2) {
    memcpy(mat+i, mat, sizeof(floatX) * prime);
  }
  for (; i < N-prime; i += prime) {
    memcpy(mat+i, mat, sizeof(floatX) * prime);
  }
  for (; i < N; i++) {
    mat[i] = mat[i-prime];
  }
#endif
#else
  cudaMemset(mat, 0, sizeof(floatX) * N);
#endif
}

/*
bool verify_matrix(floatX *matRef, floatX *matOut, int N) {
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

__global__ void verify_matrix_kernel(floatX *matRef, floatX *matOut, floatX *matI, unsigned int *error, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

/*
  if (i == 0) {
    for(int i = 0; i < 2048; i++) {
      float diff = fabs((float)matRef[i] - (float)matOut[i]);
      float ref = (float)matRef[i];
      float out = (float)matOut[i];
      float added = (float)matOut[i] + (float)matI[i];
      float added_ref = (float)matRef[i] + (float)matI[i];
      float value_I = (float)(matI[i]);
      printf("Should be %5.20f, Is %5.20f (Diff %5.7f) at %d (with I: %5.7f ==> real diff: %5.7f)\n", ref, out, diff, i, value_I, added_ref - out);
    }
    return;
  }
*/

  if (i < N) {
    float ref_with_added = (float)((floatX)(((float)matRef[i] + (float)(ENABLE_C_INPUT ? (float)matI[i] : 0.0f))));
    float diff = fabs(ref_with_added - (float)matOut[i]);

    int x_base = i % max_size;
    int y_base = (i / max_size);
    bool absmax_position = ENABLE_ABSMAX_SCALING ? (x_base % 256 == 0 && y_base % 256 == 0) : false;

    if (diff > 0.1 || absmax_position) {
      // accept result if it looks like RELU
      if ((float)matRef[i] > 0.0f || (float)matOut[i] != 0.0f || absmax_position) {
        //printf("Divergence! Should %5.20f, Is %5.20f (Diff %5.7f) at %d (with I: %5.7f)\n", ref_with_added, (float)matOut[i], diff, (int)i, (float)matI[i]);
        if (absmax_position) {
          // calculate absmax for the tile
          // (this is... not maximally efficient)
          float absmax = 0.0f;
          for (int x = 0; x < 256; x++) {
            for (int y = 0; y < 256; y++) {
              int idx = (x + x_base) + (y + y_base) * max_size;
              float ref_with_added = (float)((floatX)(((float)matRef[idx] + (float)(ENABLE_C_INPUT ? (float)matI[idx] : 0.0f))));
              absmax = max(absmax, fabsf(ref_with_added));
            }
          }
          floatX absmax_bf16 = (floatX)absmax;
          diff = fabsf((float)absmax_bf16 - (float)matOut[i]);
          if (diff > 2.0) {
            printf("absmax: %5.20f vs claimed_absmax: %5.20f (at: %d/%d)\n", (float)absmax_bf16, (float)matOut[i], (int)x_base, (int)y_base);
            *error = 1;
          }
        } else {
          if (diff > 100.0f && ((float)ref_with_added / (float)matOut[i] > 1.01f || (float)ref_with_added / (float)matOut[i] < 0.99f)) {
            printf("Divergence! Should %5.20f, Is %5.20f (Diff %5.7f) at %d\n", ref_with_added, (float)matOut[i], diff, i);
            *error = 1;
          }
        }
      }
    }
  }
}

int main() {
  get_time();
  long m = max_size, n = max_size, k = max_size;

  floatX *A = nullptr, *B = nullptr, *C = nullptr, *I = nullptr, *C_ref = nullptr;  // host matrices
  floatX *dA = nullptr, *dB = nullptr, *dC = nullptr, *dI = nullptr, *dC_ref = nullptr; // device matrices

  A = (floatX *)malloc(sizeof(floatX) * max_size * max_size);
  B = (floatX *)malloc(sizeof(floatX) * max_size * max_size);
  C = (floatX *)malloc(sizeof(floatX) * max_size * max_size);
  I = (floatX *)malloc(sizeof(floatX) * max_size * max_size);
  C_ref = (floatX *)malloc(sizeof(floatX) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(I, max_size * max_size, 0.0f);

  unsigned int* scalar_gpu;
  unsigned int scalar_host;
  cudaMalloc((void**)&scalar_gpu, sizeof(unsigned int));
  cudaCheck(cudaMemset(scalar_gpu, 0, sizeof(unsigned int)));
  cudaCheck(cudaMalloc((void **)&dA, sizeof(floatX) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(floatX) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(floatX) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dI, sizeof(floatX) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(floatX) * max_size * max_size));

  cudaCheck(cudaMemcpyAsync(dA, A, sizeof(floatX) * max_size * max_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dB, B, sizeof(floatX) * max_size * max_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dC, I, sizeof(floatX) * max_size * max_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dI, I, sizeof(floatX) * max_size * max_size, cudaMemcpyHostToDevice));

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
  for (int kernel_num : {10}) {
    printf("\nKERNEL %d\n", kernel_num);

    if (!first_run) {
      nanosleep(&ts_second, NULL); // optional rest to avoid thermal throttling between kernels
    }
    first_run = false;

    // Verify against cuBLAS. Also serves as a warmup step
    if (run_verif) {
      cudaCheck(cudaMemset(dC, 0, sizeof(floatX) * max_size * max_size));
      cudaCheck(cudaMemset(dC_ref, 0, sizeof(floatX) * max_size * max_size));
      cudaCheck(cudaMemset(scalar_gpu, 0, sizeof(unsigned int)));

      run_kernel(kernel_num, m, n, k, dA, dB, dC, dI, scalar_gpu);
      run_kernel(REFERENCE_KERNEL, m, n, k, dA, dB, dC_ref, dI, scalar_gpu);

      cudaCheck(cudaMemset(scalar_gpu, 0, sizeof(unsigned int)));
      verify_matrix_kernel<<<CEIL_DIV(m * n, 1024), 1024>>>(dC_ref, dC, dI, scalar_gpu, m * n);
      cudaMemcpy(&scalar_host, scalar_gpu, sizeof(unsigned int), cudaMemcpyDeviceToHost); // can only be async because next memcpy isn't
      printf("\n=======> Kernel %d -> VERIFICATION: %s\n\n", kernel_num, scalar_host ? "!!!!! FAILED !!!!!" : "OK");
    }

    printf("Benchmarking kernel %d - time: %d\n", kernel_num, get_time());

    // Benchmark
    cudaEventRecord(start);
    for (int j = 0; j < repeat_times; j++) {
      run_kernel(kernel_num, m, n, k, dA, dB, dC, dI, scalar_gpu);
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
  free(I);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cudaFree(scalar_gpu);
  return 0;
};