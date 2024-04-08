#ifndef BATCH_MATMUL
#define BATCH_MATMUL


void elementwise_gemm(float* A, float* B, float* output, int batch_size, int M, int N, int K){

}


void broadcasted_gemm(float* A, float* B, float* output, int batch_size, int M, int N, int K){

}


void gemm(float* A, float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc);

void elementwise_gemm_cpu(float* A, float* B, float* output, int batch_size, int M, int N, int K);
void broadcasted_gemm_cpu(float* A, float* B, float* output, int batch_size, int M, int N, int K);
  


#endif