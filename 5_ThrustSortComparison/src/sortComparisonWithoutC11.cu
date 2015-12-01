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

    std::cout<< "Time elapsed" << msg << (intmax_t)(en_time - st_time)<<std::endl;
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
}
long vec_size =atoi(argv[1]);

{
std::cout <<std::endl<<"sort test all std"<<std::endl;


start_clock();
std::vector<double> vec(vec_size);
std::for_each(vec.begin(), vec.end(), generateRandom);


std::sort(vec.begin(), vec.end());
end_clock("CPU all");
}

{
std::cout <<std::endl<<"sort test std"<<std::endl;
std::vector<double> vec(vec_size);
std::for_each(vec.begin(), vec.end(), generateRandom);



start_clock();

std::sort(vec.begin(), vec.end());

end_clock("CPU");
}

{
std::cout<<"cuda all  sort"<<std::endl;

start_clock();

thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), generateRandom);

thrust::device_vector<double>  d = hv;


thrust::sort(d.begin(), d.end());
hv = d;

end_clock("CUDA ALL");

}

{
std::cout<<"cuda sort"<<std::endl;


thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), generateRandom);


start_clock();
thrust::device_vector<double>  d = hv;


thrust::sort(d.begin(), d.end());
thrust::copy(d.begin(), d.end(), hv.begin());

end_clock("CUDA sort and copy")
}

{
std::cout<<"cuda sort only"<<std::endl;

thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), generateRandom);


thrust::device_vector<double>  d(vec_size);

start_clock();

thrust::copy(hv.begin(), hv.end(), d.begin());
thrust::sort(d.begin(), d.end());
thrust::copy(d.begin(), d.end(), hv.begin());

end_clock("CUDA sort and copy");

}
{
std::cout<<"cuda sort without copy and aloc included"<<std::endl;

thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), generateRandom);


thrust::device_vector<double>  d = hv;

start_clock();

thrust::sort(d.begin(), d.end());
end_clock("CUDA sort only");

hv = d;
}
}