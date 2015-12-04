#include <iostream>
#include <iterator>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <sys/times.h>
#include <stdint.h>


void start_clock(void);
void end_clock(char *msg);
static clock_t st_time;
static clock_t en_time;
static struct tms st_cpu;
static struct tms en_cpu;

void
start_clock()
{
    st_time = times(&st_cpu);
}


void end_clock(char *msg)
{
    en_time = times(&en_cpu);

    std::cout<< "Time on " << msg << " :"<< (intmax_t)(en_time - st_time)<<std::endl;
}

void pow_array(float *a, int power, int array_size)
{
for(int idx =0; idx<array_size; ++idx)
{
  float result=1;
  if (idx<array_size)
  {
      for(int i=0; i<power; ++i)
        result*=a[idx];
    a[idx] = result;
  }
}
}

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
  start_clock();
  int array_size = atoi(argv[1]);
  int power = atoi(argv[2]);

  float *a_host = new float[array_size]; //array on CPU
  for (int i=0; i<array_size; i++) a_host[i] = i;


  float *a_device;

//  cudaDeviceReset();

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
  delete[] a_host;
  cudaFree(a_device);

  end_clock("GPU");
  }
  {
  start_clock();
  int array_size = atoi(argv[1]);
  int power = atoi(argv[2]);

  float *a_host = new float[array_size]; //array on CPU
  for (int i=0; i<array_size; i++) a_host[i] = i;



  pow_array(a_host, power, array_size);



  // Print results
  std::ostream_iterator<float> printIterator(std::cout, "\n");
  std::copy(a_host, a_host+2, printIterator);
  std::cout<<"."<<std::endl;
  std::copy(a_host+array_size-2, a_host+array_size, printIterator);

  // Cleanup
  delete[] a_host;
  end_clock("CPU");
  }
}