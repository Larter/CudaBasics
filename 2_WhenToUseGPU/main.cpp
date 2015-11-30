#include <iostream>
#include <cuda.h>
#include <iterator>
#include <algorithm>
#include <ctime>

// Kernel that executes on the CUDA device
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



  pow_array(a_host, power, array_size);



  // Print results
  std::ostream_iterator<float> printIterator(std::cout, "\n");
  std::copy(a_host, a_host+10, printIterator);
  std::cout<<"."<<std::endl<<"."<<std::endl<<"."<<std::endl;
  std::copy(a_host+array_size-10, a_host+array_size, printIterator);

  // Cleanup
  free(a_host);
  long allEnd=clock();

  std::cout<<"Time elapsed on CPU:" <<clock()- allStart <<std::endl;
  return 0;
  }
}