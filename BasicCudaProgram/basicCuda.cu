#include <iostream>
#include <cuda.h>
#include <iterator>
#include <algorithm>


// Kernel that executes on the CUDA device
__global__ void square_array(float *a, int array_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<array_size) a[idx] = a[idx] * a[idx];
}


int main(void)
{
  int array_size = 100;
  size_t size = array_size * sizeof(float);

  float *a_host = new float[array_size]; //array on CPU
  for (int i=0; i<array_size; i++) a_host[i] = i;

  float *a_device;


  cudaMalloc((void **) &a_device, size);   // Allocate array on device
  cudaMemcpy(a_device, a_host, size, cudaMemcpyHostToDevice); //Copy data to device



  // Do calculation on device:
  int block_size = 64;
  int n_blocks = array_size/block_size + (array_size%block_size == 0 ? 0:1);

  square_array <<< n_blocks, block_size >>> (a_device, array_size);

  cudaMemcpy(a_host, a_device, sizeof(float)*array_size, cudaMemcpyDeviceToHost);//copy data back to CPU


  // Print results
  std::ostream_iterator<float> printIterator(std::cout, "\n");
  std::copy(a_host, a_host+5, printIterator);
  std::cout<<"."<<std::endl<<"."<<std::endl<<"."<<std::endl;
  std::copy(a_host+array_size-5, a_host+array_size, printIterator);

  // Cleanup
  free(a_host); cudaFree(a_device);
}