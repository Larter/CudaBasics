#include <iostream>
#include <cuda.h>
#include <iterator>
#include <algorithm>
#include <ctime>


// Kernel that executes on the CUDA device
__global__ void pow_array_gpu(float *a, int power, int array_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float result=1;
  if (idx<array_size)
  {
      for(int i=0; i<power; ++i)
        result*=a[idx];
    a[idx] = result;
  }
}


int main(int argc, char ** argv)
{

if(argc<3)
{
  std::cout<<"Please provide array size, and iteration level as arguments"<<std::endl;
  return 1;
}
{
  int allStart=clock();
  int array_size = atoi(argv[1]);
  int power = atoi(argv[2]);

  float *a_host = new float[array_size]; //array on CPU
  for (int i=0; i<array_size; i++) a_host[i] = i;


  float *a_device;

  long mallocStart= clock();
  cudaMalloc((void **) &a_device, array_size*sizeof(float));   // Allocate array on device

  long memcpyStart=clock();
  cudaMemcpy(a_device, a_host, array_size*sizeof(float), cudaMemcpyHostToDevice);


  // Do calculation on device:
  int block_size = 256;
  int n_blocks = array_size/block_size + (array_size%block_size == 0 ? 0:1);

  pow_array_gpu <<< n_blocks, block_size >>> (a_device, power, array_size);
  // Retrieve result from device and store it in host array


  cudaMemcpy(a_host, a_device, sizeof(float)*array_size, cudaMemcpyDeviceToHost);

  // Print results
  std::ostream_iterator<float> printIterator(std::cout, "\n");
  std::copy(a_host, a_host+2, printIterator);
  std::cout<<"."<<std::endl;
  std::copy(a_host+array_size-2, a_host+array_size, printIterator);

  // Cleanup
  free(a_host); cudaFree(a_device);

  long allEnd=clock();

  std::cout<<"Time elapsed GPU :" <<clock()- allStart <<std::endl;
  }
}