
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>



void fillmatrix(double** matrix, int *n);
void printmatrix(double* matrix, int n);
cudaError_t countDeter(double* matrix, int n, double * determinant);
__global__ void addRows(double *matrix, int* i);
__global__ void multiplyDet(double *matrix, double *determinant,int* n);
int main()
{

	double determinant=1;
    cudaError_t cudaStatus;
	int n,i;
	double* matrix=NULL;
	for(i=0;i<2;i++){
		determinant=1;
	fillmatrix(&matrix, &n);

	//printmatrix(matrix,n);
	long start=clock();
	cudaStatus=countDeter(matrix,n,&determinant);
	long stop=clock();

	printf("new matrix after %ldl\n", stop-start);
	//printmatrix(matrix,n);
	printf("determinant equals : %f \n",determinant);



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	}
    return 0;
}

void fillmatrix(double **matrix, int *n){
	int nn; //n^2
	int i;//for loop
	*n=10000;
	nn=*n**n;
	*matrix=(double*)malloc( nn*sizeof(double));//allocating memory

	for(i=0;i<nn;i++){
		(*matrix)[i]=rand()%100; //filling matrix
	}
}


void printmatrix(double* matrix, int n){

	/*function prints matrix */
	int i;
	for(i=0;i<n*n;i++){
		printf("%.0f%c",matrix[i], (i+1)%n? ' ': '\n');
	}
}

cudaError_t countDeter(double *matrix, int n, double *determinant){

	double *d_matrix=NULL;
	double *d_determinant=NULL;
	int *d_i=NULL,i;
	cudaError_t cudaStatus;
	cudaStatus=cudaMalloc((void**)&d_matrix,n*n*sizeof(double));//allocating memory for matrix on gpu
	if(cudaStatus!=cudaSuccess){
		printf("Error in allocating memory");
		return cudaStatus;
	}
	cudaStatus=cudaMalloc((void**)&d_determinant,sizeof(double));//allocating memory for determinant
	if(cudaStatus!=cudaSuccess){
		printf("Error in allocating memory");
		return cudaStatus;
	}

	cudaStatus=cudaMalloc((void**)&d_i, sizeof(int));// allocating memory for index
	if(cudaStatus!=cudaSuccess){
		printf("Error in allocating memory");
		return cudaStatus;
	}

	cudaStatus=cudaMemcpy(d_matrix,matrix,n*n*sizeof(double), cudaMemcpyHostToDevice);

	if(cudaStatus!=cudaSuccess){
		printf("Error in copying");
		return cudaStatus;
	}

	cudaStatus=cudaMemcpy(d_determinant, determinant,sizeof(double), cudaMemcpyHostToDevice);

	if(cudaStatus!=cudaSuccess){
		printf("Error in copying");
		return cudaStatus;
	}

	//Here starts the coding
	long start=clock();
	for(i=0;i<n;i++){

		cudaStatus=cudaMemcpy(d_i, &i,sizeof(int), cudaMemcpyHostToDevice);
		if(cudaStatus!=cudaSuccess){
			printf("Error in copying");
			return cudaStatus;
		}

		addRows<<< n-i-1,n-i>>>(d_matrix,d_i);
	}
	long stop=clock();

	printf("czas samej petli:%ld\n",stop-start);


	i=n;
	cudaStatus=cudaMemcpy(d_i, &i,sizeof(int), cudaMemcpyHostToDevice);
	if(cudaStatus!=cudaSuccess){
		printf("Error in copying");
		return cudaStatus;
	}
	//multiplyDet <<< 1 , 1 >>> (matrix,d_determinant, d_i);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		return cudaStatus;
    }

	cudaStatus = cudaMemcpy(determinant , d_determinant, sizeof(double), cudaMemcpyDeviceToHost);

	if(cudaStatus!=cudaSuccess){
		printf("Error in copying1");
		//return cudaStatus;
	}

	cudaStatus=cudaMemcpy(matrix, d_matrix,n*n*sizeof(double), cudaMemcpyDeviceToHost);

	if(cudaStatus!=cudaSuccess){
		printf("Error in copying");
		return cudaStatus;
	}
	for(i=0;i<n;i++){
		*determinant=(*determinant)*matrix[i*(n)+i];
	}

	cudaFree(d_matrix);
	cudaFree(d_determinant);
	cudaFree(d_i);


	return cudaStatus;


}

__global__ void addRows(double *matrix, int *d_i){
	int i=*d_i;
	int n=blockDim.x+i;
	int id= n*(blockIdx.x+i+1) + threadIdx.x+i;
	__shared__ double multiplier;

	if(threadIdx.x==0){
		multiplier=matrix[n*(blockIdx.x+1+i)+i]/matrix[n*i+i];
	}
   __syncthreads();

	matrix[id]-=matrix[n*i+threadIdx.x+i]*multiplier;
}

__global__ void multiplyDet(double *matrix, double *determinant,int* n){

	int i;
	int nn=*n;
	for(i=0;i<nn;i++){
		*determinant=(*determinant)*matrix[i*(nn)+i];
	}
}



/*
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
*/