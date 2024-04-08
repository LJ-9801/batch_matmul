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

#define MIN -2
#define MAX 2

#define RANGE (MAX + 1 - MIN) + MAX

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
    A[i] = rand() % RANGE;
  }

  for (int i = 0; i< B_size; i++){
    B[i] = rand() % RANGE;
  }
  
  cudaMalloc((void**)&A_dev, sizeof(float)*A_size);
  cudaMalloc((void**)&B_dev, sizeof(float)*B_size);
  cudaMalloc((void**)&C_dev, sizeof(float)*C_size);

  cudaMemcpy((void*)A_dev, A, sizeof(float)*A_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)B_dev, B, sizeof(float)*B_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)C_dev, C, sizeof(float)*C_size, cudaMemcpyHostToDevice);

  std::cout << "CPU ops" << std::endl;
  elementwise_gemm_cpu(A, B, C, BATCH_SIZE, M, N, K);
  std::cout << "GPU ops" << std::endl;
  elementwise_gemm(A_dev, B_dev, C_dev, BATCH_SIZE, M, N, K);


  delete[] A;
  delete[] B;
  delete[] C;

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);

  return 0;
}