#include <iostream>
#include "batch_matmul.cuh"

#define EQUAL

#ifdef EQUAL
  // equal is when two tensor has the batches of matrices
  // let's say 
  // Tensor A has         shape[3, 4, M, K]
  // Tensor B has         shape[3, 4, K, N]
  // Tensor C will output shape[3, 4, M, N]
#else
  // this is when one tensor has an arbitary shape
  // and the other is a matrix of 2 dimension 
  // let's say
  // Tensor A has         shape[3, 4, M, K]
  // Tensor B has         shape      [K, M]
  // Tensor C will output shape[3, 4, M, N] 
#endif

// put your param here
#define BATCH_SIZE 3*4
#define M 1024
#define N 1024
#define K 1024

// you can set your tolarance for comparing with your result with groud truth
#define TOL 0.005

#define MIN -2
#define MAX 2

#define RANGE (MIN + (MAX - MIN) * ((float)rand() / RAND_MAX))

int main(){

  size_t A_size = BATCH_SIZE*M*K;
#ifdef EQUAL
  std::cout << "Doing elementwise matmul" << std::endl;
  size_t B_size = BATCH_SIZE*K*N;
#else
  std::cout << "Doing broadcasted matmul" << std::endl;
  size_t B_size = K*N;
#endif
  size_t C_size = BATCH_SIZE*M*N; 

  float* A = new float[A_size];
  float* B = new float[B_size];
  float* C = new float[C_size];

  float* A_dev = nullptr;
  float* B_dev = nullptr;
  float* C_dev = nullptr; 

  // fill in a random number from range MIN to MAX
  for (int i = 0; i < A_size; i++){
    A[i] = RANGE;
  }

  for (int i = 0; i< B_size; i++){
    B[i] = RANGE;
  }

  cudaMalloc((void**)&A_dev, sizeof(float)*A_size);
  cudaMalloc((void**)&B_dev, sizeof(float)*B_size);
  cudaMalloc((void**)&C_dev, sizeof(float)*C_size);

  cudaMemcpy((void*)A_dev, (void*)A, sizeof(float)*A_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)B_dev, (void*)B, sizeof(float)*B_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)C_dev, (void*)C, sizeof(float)*C_size, cudaMemcpyHostToDevice);

  std::cout << "CPU ops" << std::endl;
  #ifdef EQUAL
    elementwise_gemm_cpu(A, B, C, BATCH_SIZE, M, N, K);
  #else
    broadcasted_gemm_cpu(A, B, C, BATCH_SIZE, M, N, K);
  #endif

  std::cout << "GPU ops" << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  #ifdef EQUAL
    elementwise_gemm(A_dev, B_dev, C_dev, BATCH_SIZE, M, N, K);
  #else
    broadcasted_gemm(A_dev, B_dev, C_dev, BATCH_SIZE, M, N, K);
  #endif
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  float* C_verify = new float[C_size];
  cudaMemcpy((void*)C_verify, (void*)C_dev, sizeof(float)*C_size, cudaMemcpyDeviceToHost);

  std::cout << (verify(C, C_verify, C_size, TOL)? "OK" : "not OK") << std::endl;
  std::cout << "Time elapsed: " << milliseconds << "ms" << std::endl;

  delete[] A;
  delete[] B;
  delete[] C;

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);

  return 0;
}