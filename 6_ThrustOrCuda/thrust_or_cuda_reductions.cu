#include <iostream>
#include <cuda.h>
#include <iterator>
#include <algorithm>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <sys/times.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>



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

    std::cout<< "Algoritm run with: " << msg << std::endl<< " Time elapsed:"<< (intmax_t)(en_time - st_time)<<std::endl;
}


__global__ void reduce_max_kernel(float *d_out, const float *d_logLum, int size) {

    int tid         = threadIdx.x;                              // Local thread index
    int myId        = blockIdx.x * blockDim.x + threadIdx.x;    // Global thread index

    extern __shared__ float temp[];

    // --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
    temp[tid] = (myId < size) ? d_logLum[myId] : -10000000;

    // --- Your solution
    // if (myId < size) { temp[tid] = d_logLum[myId]; } else { temp[tid] = d_logLum[tid]; }

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s) { temp[tid] = fmaxf(temp[tid], temp[tid + s]); }
        // --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

    // --- Your solution
    //for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    //    if (tid < s) { if (myId < size) { temp[tid] = fmaxf(d_logLum[myId + s], d_logLum[myId]); } else { temp[tid] = d_logLum[tid]; } }
    //    __syncthreads(); 
    //}

    if (tid == 0) {
        d_out[blockIdx.x] = temp[0];
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


struct power_operator
{
  int power;
  float result;
  power_operator(int p) :power(p) {};
__device__ __host__
  float operator()(const float& value)
  {
  result=1;
      for(int i=0; i<power; ++i)
        result*=value;
    return result;
  }

};

int main(int argc, char ** argv)
{

if(argc<3)
{
  std::cout<<"Please provide array size, and iteration level as arguments"<<std::endl;
  return 1;
}
  int array_size = atoi(argv[1]);
  int power = atoi(argv[2]);

{
  cudaDeviceReset();
  start_clock();
  float *a_host = new float[array_size]; //array on CPU
  for (int i=0; i<array_size; i++) a_host[i] = i;


  float *a_device;
  float *reduce_helper;
//  cudaDeviceReset();

  int reduce_size = 256;

  cudaMalloc((void **) &a_device, array_size*sizeof(float));   // Allocate array on device
  cudaMalloc((void **) &reduce_helper, reduce_size*sizeof(float));   // Allocate array on device

  cudaMemcpy(a_device, a_host, array_size*sizeof(float), cudaMemcpyHostToDevice);



  // Do calculation on device:
  int block_size = 7;

  int n_blocks = array_size/block_size + (array_size%block_size == 0 ? 0:1);

  pow_array_gpu <<< n_blocks, block_size >>> (a_device, power, array_size);


  reduce_max_kernel<<< reduce_size, array_size/reduce_size>>> (reduce_helper, a_device, array_size);
  reduce_max_kernel<<< 1, reduce_size>>>(a_device, reduce_helper, reduce_size);
  // Retrieve result from device and store it in host array


  cudaMemcpy(a_host, a_device, sizeof(float), cudaMemcpyDeviceToHost);

  // Print results
  std::cout<< "Result is "<< *a_host <<std::endl;
  // Cleanup
  free(a_host); cudaFree(a_device); cudaFree(reduce_helper);

  long allEnd=clock();

  end_clock("CUDA");
  }
  {
  cudaDeviceReset();
  int allStart=clock();
    {
      thrust::host_vector<float> v_host(array_size);
      for (int i=0; i<array_size; i++) v_host[i] = i;

      thrust::device_vector<float> v_device= v_host;

      float result= thrust::transform_reduce(v_device.begin(), v_device.end(), power_operator(power), 0, thrust::maximum<float>());
      std::cout<<"Result is " <<result <<std::endl;
    }
  long allEnd=clock();

  end_clock("THRUST");
  }
}