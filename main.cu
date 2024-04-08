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

#define BATCH_SIZE 3*4
#define M 1024
#define N 1024
#define K 1024


int main(){

  float* A = new float[M*N];
  float* B = new float[K*M];
  float* C = new float[M*N];

  for (int i = 0; i < M*N; i++){
    A[i] = rand();
  }



  return 0;
}