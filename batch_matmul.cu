#include "batch_matmul.cuh"

// ======================================
// *************** cpu ops **************
// ======================================
void gemm(float* A, float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc){
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      float sum = 0;
      for(int k = 0; k < K; j++){
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
    gemm(A_start, B_start, C_start, M, N, K, K, M, N);
  }
}


void broadcasted_gemm_cpu(float* A, float* B, float* output, int batch_size, int M, int N, int K){
  for(int i = 0; i<batch_size; i++){
    float* A_start = A + batch_size * M * K;
    float* B_start = B;
    float* C_start = output + batch_size * M * N;
    gemm(A_start, B_start, C_start, M, N, K, K, M, N);
  } 
}

// ======================================
// *************** gpu ops **************
// ======================================