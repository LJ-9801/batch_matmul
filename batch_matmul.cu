#include "batch_matmul.cuh"

// =======================================
// *************** cpu ops ***************
// =======================================

void gemm_cpu(float* A, float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc){
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      float sum = 0;
      for(int k = 0; k < K; k++){
        sum += A[i*lda+k] * B[k*ldb+j];
      }
      C[i*ldc+j] = sum;
    }
  }
}

void elementwise_gemm_cpu(float* A, float* B, float* output, int batch_size, int M, int N, int K){
  for(int i = 0; i<batch_size; i++){
    float* A_start = A + batch_size * M * K;
    float* B_start = B + batch_size * K * N;
    float* C_start = output + batch_size * M * N;
    gemm_cpu(A_start, B_start, C_start, M, N, K, K, M, N);
  }
}


void broadcasted_gemm_cpu(float* A, float* B, float* output, int batch_size, int M, int N, int K){
  for(int i = 0; i<batch_size; i++){
    float* A_start = A + batch_size * M * K;
    float* B_start = B;
    float* C_start = output + batch_size * M * N;
    gemm_cpu(A_start, B_start, C_start, M, N, K, K, M, N);
  } 
}

// =======================================
// *************** gpu ops ***************
// =======================================

__global__ void gemm_gpu(float* A, float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc){
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0;

  if(i < M && j < N){
    for(int k = 0; k < K; k++){
      sum += A[i * lda + k] * B[k * ldb + j];
    } 
  }

  C[i * ldc + j] = sum;
}


void elementwise_gemm(float* A, float* B, float* output, int batch_size, int M, int N, int K){
  const dim3 block(32, 32);
  const dim3 grid((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  for(int i = 0; i<batch_size; i++){
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float* A_start = A + batch_size * M * K;
    float* B_start = B + batch_size * K * N;
    float* C_start = output + batch_size * M * N;
    gemm_gpu<<<grid, block, 0, stream>>>(A_start, B_start, C_start, M, N, K, K, M, N);
  }
}


void broadcasted_gemm(float* A, float* B, float* output, int batch_size, int M, int N, int K){
  const dim3 block(32, 32);
  const dim3 grid((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  for(int i = 0; i<batch_size; i++){
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float* A_start = A + batch_size * M * K;
    float* B_start = B;
    float* C_start = output + batch_size * M * N;
    gemm_gpu<<<grid, block, 0, stream>>>(A_start, B_start, C_start, M, N, K, K, M, N);
  }
}