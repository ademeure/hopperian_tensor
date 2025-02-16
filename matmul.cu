#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <assert.h>

//#define FP8

#ifdef FP8
#include <cuda_fp8.h>
typedef __nv_fp8_e4m3 floatX;
#define WGMMA_INSTRUCTION "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3"
constexpr auto CU_TENSOR_FLOATX = CU_TENSOR_MAP_DATA_TYPE_UINT8;
#define MAX_DIFF_ABS 8.0f
#define MAX_DIFF_REL 1.05f
#else
typedef __nv_bfloat16 floatX;
#define WGMMA_INSTRUCTION "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16"
constexpr auto CU_TENSOR_FLOATX = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
#define MAX_DIFF_ABS 0.01f
#define MAX_DIFF_REL 1.001f
#endif

typedef __nv_bfloat16 floatP;
#define CUBLAS_FLOATP CUDA_R_16BF
constexpr auto CU_TENSOR_FLOATP = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;

constexpr bool ENABLE_C_INPUT = false;
constexpr float ENABLE_ABSMAX_SCALING = 0.0f;
constexpr bool REDUCE_SHARED_CONFLICTS = true;

#define ENABLE_CUBLAS
#define ENABLE_RANDOM
#define ENABLE_TRUE_RANDOM
#define SLEEP_BETWEEN_KERNELS_SEC 0 // optional rest to avoid thermal throttling between kernels
constexpr bool RUN_VERIF = true;
constexpr int metadata_size = 16384; // big enough so it's spread over both L2 sides
constexpr int max_size = 16384;
constexpr int prime = 3719;
int repeat_times = 400;

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
void runCublasGemmBF16(int M, int N, int K, floatP *A, floatP *B, floatP *C) {
  float alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUBLAS_FLOATP,
    K, B, CUBLAS_FLOATP, K, &beta, C, CUBLAS_FLOATP, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS error: %d\n", status);
    exit(1);
  }
}
#endif

void run_kernel(int kernel_num, int M, int N, int K, floatX *A, floatX *B, floatP *C, floatP *I, unsigned int* metadata_gpu) {
  switch (kernel_num) {
    case 0:
#ifdef ENABLE_CUBLAS
      runCublasGemmBF16(M, N, K, (floatP*)A, (floatP*)B, C);
#endif
      break;
    case 10:
      runKernel10(M, N, K, A, B, C, I, metadata_gpu);
      break;
  }
}

void randomize_matrix(floatP *mat, int N, float scale=1.0f) {
  if (scale == 0.0f) {
    for (int i = 0; i < N; i++) {
      mat[i] = (floatP)(i*1000);
    }
    return;
  }
#ifdef ENABLE_RANDOM
  std::normal_distribution<float> distribution(0, scale);
#ifdef ENABLE_TRUE_RANDOM
  for (int i = 0; i < N; i++) {
    mat[i] = (floatP)((floatX)(distribution(generator) + 0.01f));
  }
#else
  int i = 0;
  for (; i < prime; i++) {
    mat[i] = (floatP)((floatX)distribution(generator));
  }
  for (int multiplier = 1; i < N-(prime * multiplier); i += prime * multiplier, multiplier *= 2) {
    memcpy(mat+i, mat, sizeof(floatP) * prime);
  }
  for (; i < N-prime; i += prime) {
    memcpy(mat+i, mat, sizeof(floatP) * prime);
  }
  for (; i < N; i++) {
    mat[i] = mat[i-prime];
  }
#endif
#else
  cudaMemset(mat, 0, sizeof(floatP) * N);
#endif
}

__global__ void verify_matrix_kernel(floatP *matRef, floatP *matOut, floatP *matI, unsigned int *error, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    float ref_with_added = (float)((floatP)(((float)matRef[i] + (float)(ENABLE_C_INPUT ? (float)matI[i] : 0.0f))));
    float diff = fabs(ref_with_added - (float)matOut[i]);

    int x_base = i % max_size;
    int y_base = (i / max_size);

    if (diff > 0.1) {
      // (hack) accept result if it looks like RELU
      if ((float)matRef[i] > 0.0f || (float)matOut[i] != 0.0f) {
        if (diff > MAX_DIFF_ABS && ((float)ref_with_added / (float)matOut[i] > MAX_DIFF_REL || (float)ref_with_added / (float)matOut[i] < (1.0f/MAX_DIFF_REL))) {
          if(!*error) {
            printf("Divergence! Should %5.20f, Is %5.20f (Diff %5.7f) at %d\n", ref_with_added, (float)matOut[i], diff, i);
            *error = 1;
          }
        }
      }
    } else if (i < 140) {
      //: %5.20f, Is %5.20f (Diff %5.7f) at %d\n", ref_with_added, (float)matOut[i], diff, i);
    }
  }
}

__global__ void copy_to_floatX(floatP *input, floatX *output, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    output[i] = (floatX)input[i];
  }
}

int main() {
  get_time();
  long m = max_size, n = max_size, k = max_size; // TODO: doesn't work if not all the same yet, need to debug

  floatP *A = nullptr, *B = nullptr, *C = nullptr, *I = nullptr, *C_ref = nullptr;  // host matrices
  floatP *dA = nullptr, *dB = nullptr, *dC = nullptr, *dI = nullptr, *dC_ref = nullptr; // device matrices
  floatX *dA_X = nullptr,*dB_X= nullptr, *dC_X = nullptr, *dI_X;

  A = (floatP *)malloc(sizeof(floatP) * m * k);
  B = (floatP *)malloc(sizeof(floatP) * n * k);
  C = (floatP *)malloc(sizeof(floatP) * m * n);
  I = (floatP *)malloc(sizeof(floatP) * m * n);
  C_ref = (floatP *)malloc(sizeof(floatP) * m * n);

  randomize_matrix(A, m * k);
  randomize_matrix(B, n * k);
  randomize_matrix(I, m * n, 0.0f);

  unsigned int* metadata_gpu;
  unsigned int scalar_host;
  cudaMalloc((void**)&metadata_gpu, metadata_size);
  cudaCheck(cudaMemset(metadata_gpu, 0, metadata_size));
  cudaCheck(cudaMalloc((void **)&dA, sizeof(floatP) * m * k));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(floatP) * n * k));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(floatP) * m * n));
  cudaCheck(cudaMalloc((void **)&dI, sizeof(floatP) * m * n));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(floatP) * m * n));

  cudaCheck(cudaMalloc((void **)&dA_X, sizeof(floatX) * m * k));
  cudaCheck(cudaMalloc((void **)&dB_X, sizeof(floatX) * n * k));
  cudaCheck(cudaMalloc((void **)&dC_X, sizeof(floatX) * m * n));
  cudaCheck(cudaMalloc((void **)&dI_X, sizeof(floatX) * m * n));

  cudaCheck(cudaMemcpyAsync(dA, A, sizeof(floatP) * m * k, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dB, B, sizeof(floatP) * n * k, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dC, I, sizeof(floatP) * m * n, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dI, I, sizeof(floatP) * m * n, cudaMemcpyHostToDevice));

  copy_to_floatX<<<CEIL_DIV(m * k, 1024), 1024>>>(dA, dA_X, m * k);
  copy_to_floatX<<<CEIL_DIV(n * k, 1024), 1024>>>(dB, dB_X, n * k);
  copy_to_floatX<<<CEIL_DIV(m * n, 1024), 1024>>>(dC, dC_X, m * n);
  copy_to_floatX<<<CEIL_DIV(m * n, 1024), 1024>>>(dI, dI_X, m * n);

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
  for (int kernel_num : {10, 10, 10, 10, 10, 10, 10, 0, 0}) {
    printf("\nKERNEL %d\n", kernel_num);

    if (!first_run && SLEEP_BETWEEN_KERNELS_SEC) {
      nanosleep(&ts_second, NULL); // optional rest to avoid thermal throttling between kernels
    }
    first_run = false;

#ifdef ENABLE_CUBLAS
    // Verify against cuBLAS. Also serves as a warmup step
    if (run_verif) {
      cudaCheck(cudaMemset(dC, 0, sizeof(floatP) * m * n));
      cudaCheck(cudaMemset(dC_ref, 0, sizeof(floatP) * m * n));
      cudaCheck(cudaMemset(metadata_gpu, 0, metadata_size));

      run_kernel(kernel_num, m, n, k, dA_X, dB_X, dC, dI, metadata_gpu);
      runCublasGemmBF16(m, n, k, dA, dB, dC_ref);

      cudaCheck(cudaMemset(metadata_gpu, 0, metadata_size));
      verify_matrix_kernel<<<CEIL_DIV(m * n, 1024), 1024>>>(dC_ref, dC, dI, metadata_gpu, m * n);
      cudaMemcpy(&scalar_host, metadata_gpu, sizeof(unsigned int), cudaMemcpyDeviceToHost); // can only be async because next memcpy isn't
      printf("\n=======> Kernel %d -> VERIFICATION: %s\n\n", kernel_num, scalar_host ? "!!!!! FAILED !!!!!" : "OK");
    }
#endif

    printf("Benchmarking kernel %d - time: %d\n", kernel_num, get_time());

    // Benchmark
    cudaEventRecord(start);
    for (int j = 0; j < repeat_times; j++) {
      run_kernel(kernel_num, m, n, k, dA_X, dB_X, dC, dI, metadata_gpu);
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
  cudaFree(dA_X);
  cudaFree(dB_X);
  cudaFree(dC_X);
  cudaFree(dI_X);
  cudaFree(metadata_gpu);
  return 0;
};