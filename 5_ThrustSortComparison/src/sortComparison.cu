#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <iterator>
#include <thrust/copy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <thrust/sort.h>


int main()
{

auto vec_size =500000000;
auto start = std::chrono::high_resolution_clock::now();
auto end = std::chrono::high_resolution_clock::now();

{
std::cout <<std::endl<<"sort test all  std"<<std::endl;



auto start = std::chrono::high_resolution_clock::now();
std::vector<double> vec(vec_size);
std::for_each(vec.begin(), vec.end(), [](double & d){d=rand();});


std::sort(vec.begin(), vec.end());

auto end = std::chrono::high_resolution_clock::now();


 std::cout << "STD sort took  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms.\n";
}
{
std::cout <<std::endl<<"sort test std"<<std::endl;
std::vector<double> vec(vec_size);
std::for_each(vec.begin(), vec.end(), [](double & d){d=rand();});



auto start = std::chrono::high_resolution_clock::now();


std::sort(vec.begin(), vec.end());

auto end = std::chrono::high_resolution_clock::now();


 std::cout << "STD sort took  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms.\n";
}

{
std::cout<<"cuda all  sort"<<std::endl;





start = std::chrono::high_resolution_clock::now();
thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), [](double & d){d=rand();});

thrust::device_vector<double>  d = hv;


thrust::sort(d.begin(), d.end());
hv = d;

end = std::chrono::high_resolution_clock::now();


 std::cout << "Cuda sort took   "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms.\n";
}

{
std::cout<<"cuda sort"<<std::endl;


thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), [](double & d){d=rand();});



start = std::chrono::high_resolution_clock::now();
thrust::device_vector<double>  d = hv;


thrust::sort(d.begin(), d.end());
hv = d;

end = std::chrono::high_resolution_clock::now();


 std::cout << "Cuda sort took   "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms.\n";
}

{
std::cout<<"cuda sort without aloc included"<<std::endl;

thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), [](double & d){d=rand();});


thrust::device_vector<double>  d(vec_size);


start = std::chrono::high_resolution_clock::now();

thrust::copy(hv.begin(), hv.end(), d.begin());

thrust::sort(d.begin(), d.end());
end = std::chrono::high_resolution_clock::now();

hv = d;

 std::cout << "Cuda sort took   "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms.\n";
}
{
std::cout<<"cuda sort without copy and aloc included"<<std::endl;

thrust::host_vector<double> hv(vec_size);
std::for_each(hv.begin(), hv.end(), [](double & d){d=rand();});


thrust::device_vector<double>  d = hv;

start = std::chrono::high_resolution_clock::now();


thrust::sort(d.begin(), d.end());
end = std::chrono::high_resolution_clock::now();

hv = d;

 std::cout << "Cuda sort took   "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms.\n";
}
}