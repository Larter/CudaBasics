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


int main()
{

thrust::host_vector<int> host_vec(5);
for(int z = 0; z <host_vec.size();++z)
       host_vec[z]=z;

std::cout<<"Host vector before"<< std::endl;
std::copy(host_vec.begin(), host_vec.end(), std::ostream_iterator<int>(std::cout, " "));
std::cout<<std::endl;

thrust::device_vector<int> dev_vec = host_vec;

thrust::negate<int> op;
thrust::transform(dev_vec.begin(), dev_vec.end(),dev_vec.begin(),op);

thrust::copy(dev_vec.begin(), dev_vec.end(), host_vec.begin());


std::cout<<"Host vector After"<< std::endl;
std::copy(host_vec.begin(), host_vec.end(), std::ostream_iterator<int>(std::cout, " "));
std::cout<<std::endl;

}