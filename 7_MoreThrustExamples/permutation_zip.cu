#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

struct int_float_multi_sum_operator
{
	__device__ __host__
	thrust::tuple<int,float> operator()(thrust::tuple<int,float> prev_sum, const thrust::tuple<int,float> & current )
	{
		thrust::get<0>(prev_sum)+=thrust::get<0>(current)*thrust::get<1>(current);
		return prev_sum;
	}
};


int main(int argc, char const *argv[])
{

	thrust::device_vector<int> ints(300);

	for(int i=0; i < ints.size();++i)
	{
		ints[i]=i;
	}

	thrust::device_vector<float> floats(300);
	for(int i=0; i < floats.size();++i)
	{
		floats[i]=floats.size()-i;
	}



	float sum = thrust::get<0>(thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(ints.begin(), floats.begin())), 
								thrust::make_zip_iterator(thrust::make_tuple(ints.end(), floats.end())),
								0, int_float_multi_sum_operator()));

	std::cout<<sum<<std::endl;
	return 0;
}