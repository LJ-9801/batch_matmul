#!/usr/bin/bash

echo "Compiling"
nvcc batch_matmul.cu main.cu -Xcompiler -fopenmp -O3 -o main
echo "Running"
./main