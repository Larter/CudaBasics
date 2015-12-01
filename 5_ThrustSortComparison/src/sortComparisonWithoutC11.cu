#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <iterator>
#include <thrust/copy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform.h>
#include <algorithm>
#include <vector>
#include <thrust/sort.h>
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

    std::cout<< "Sort type : " << msg << std::endl<< " Time elapsed:"<< (intmax_t)(en_time - st_time)<<std::endl;
}




void generateRandom(double & i)
{
       i = rand();
}


int main(int argc, char ** argv)
{

if(argc<2)
{
       std::cout<<"Please provide size as argument"<<std::endl;
       return 1;
}
long vec_size =atoi(argv[1]);

{
start_clock();
std::vector<double> vec(vec_size);
std::for_each(vec.begin(), vec.end(), generateRandom);


std::sort(vec.begin(), vec.end());
end_clock("CPU all");
}

{
std::vector<double> vec(vec_size);
std::for_each(vec.begin(), vec.end(), generateRandom);



start_clock();

std::sort(vec.begin(), vec.end());

end_clock("CPU sort only");
}

{
cudaDeviceReset();
start_clock();

thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), generateRandom);

thrust::device_vector<double>  d = hv;


thrust::sort(d.begin(), d.end());
hv = d;

end_clock("thrust ALL");

}

{

cudaDeviceReset();
thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), generateRandom);


start_clock();
thrust::device_vector<double>  d = hv;


thrust::sort(d.begin(), d.end());
thrust::copy(d.begin(), d.end(), hv.begin());

end_clock("Thrust sort and copy and alloc");
}

{
cudaDeviceReset();
thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), generateRandom);


thrust::device_vector<double>  d(vec_size);

start_clock();

thrust::copy(hv.begin(), hv.end(), d.begin());
thrust::sort(d.begin(), d.end());
thrust::copy(d.begin(), d.end(), hv.begin());

end_clock("thrust sort and copy");

}
{
cudaDeviceReset();
thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), generateRandom);


thrust::device_vector<double>  d = hv;

start_clock();

thrust::sort(d.begin(), d.end());
end_clock("thrust sort only");

hv = d;
}
}