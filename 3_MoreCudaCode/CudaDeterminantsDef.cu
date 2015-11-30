#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define VALUESMAX 100
#define BMARK -1
#define SIZE 2
#define PRINTMATRIX 1
#define PRINTPERM 0
#define SINGLETONS 0 //1 if singletons 0 if inversions
#define RAND 5

__device__ void nextPermutationBlock(long double *matrix, long double *permutations, bool *usedValues, int n, long double value, int parametr, int *fractionNumber, int level);
__global__ void permutations(long double *matrix,long double *permutationValues);
__global__ void addPermutations(long double *determinant, long double *permutations, int *n);

int main(){

	long double *matrix, *d_matrix, *d_permutationValues, *d_determinant, determinant;
	int n=SIZE, *d_n;
	cudaError_t cudaStatus;

	matrix=(long double*)malloc(sizeof(long double)*n*n);//alocating matrix

	for(int i=0; i< n*n; i++){
		matrix[i]=rand()%RAND;
		if(PRINTMATRIX==1){
			printf("%Lf ",matrix[i]);
			if(!((i+1)%n)){
			    printf("\n");
			}//if
		}//PRINTMATRIX
	}//for


	cudaStatus=cudaMalloc((void**)&d_matrix, n*n*sizeof(long double)); /*allocating matrix memory on gpu*/

	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"Error in allocating memory\n");
		return cudaSuccess;
	}

	cudaStatus=cudaMalloc((void**)&d_n,sizeof(int));

	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"Error in allocating memory\n");
		return cudaStatus;
	}


	cudaStatus=cudaMalloc((void**)&d_permutationValues, (n-1)*n*sizeof(long double)); /*allocating matrix memory on gpu*/

	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"Error in allocating memory\n");
		return cudaSuccess;
	}

	cudaStatus=cudaMalloc((void**)&d_determinant, sizeof(long double)); /*allocating matrix memory on gpu*/

	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"Error in allocating memory\n");
		return cudaSuccess;
	}


	cudaStatus=cudaMemcpy(d_matrix,matrix,n*n*sizeof(long double),cudaMemcpyHostToDevice);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"Error in copying matrix memory, %d\n");
		return cudaStatus;
	}

	cudaStatus=cudaMemcpy(d_n,&n,sizeof(int),cudaMemcpyHostToDevice);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"Error in copying matrix memory\n");
		return cudaStatus;
	}









	permutations<<<n,n-1>>>(d_matrix,d_permutationValues);


	//copying memory

	addPermutations<<<1,1>>>(d_determinant,d_permutationValues,d_n);

	cudaStatus=cudaMemcpy(&determinant,d_determinant,sizeof(long double),cudaMemcpyDeviceToHost);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"Error in copying matrix memory, %d\n");
		return cudaStatus;
	}

	printf("Determinant equals: %Lf \n", determinant);


}




__device__ void nextPermutationBlock(long double *matrix, long double *permutations, bool *usedValues, int n, long double value, int parametr, int *fractionNumber, int level){


	if(level==n){
		if(SINGLETONS==1){
			value*=pow((float)BMARK,n-parametr);
		}
		else{
			value*=pow((float)BMARK,parametr);
		}
		*permutations+=value;

	}
	else{
		int addValue=-1;
		for(int i=0; i<n; i++){
			if(usedValues[i]==true){
				usedValues[i]=false;
				if(SINGLETONS==1){
					if(i==level){
						addValue=1;
					}
					else{
						addValue=0;
					}
				}
				else{//inversions
					addValue++;
				}


				nextPermutationBlock(matrix,permutations,usedValues,n,value*matrix[level*n+i],parametr+addValue,fractionNumber,level+1);
				usedValues[i]=true;
			}


		}

	}
}


__global__ void addPermutations(long double *determinant, long double *permutations, int *n){

	int nn=*n**n-1;
	*determinant=0;
	for(int i=0;i<nn;i++){
		*determinant+=permutations[i];
	}
}

__global__ void permutations(long double *matrix,long double *permutationValues){

	int *fractionNumber[1]={0};
	int n=gridDim.x;
	int inversions;
	bool usedValues[VALUESMAX];
	double result=1;
	for(int i=0; i<blockDim.x+1;i++){
		usedValues[i]=true;
	}
	usedValues[blockIdx.x]=false;
	usedValues[threadIdx.x]=false;
	result*=matrix[blockIdx.x];
	result*=matrix[n+threadIdx.x];

	inversions=blockIdx.x+threadIdx.x;
	if(blockIdx.x<threadIdx.x){
		inversions--;
	}

	nextPermutationBlock(matrix, &permutationValues[blockIdx.x*n+threadIdx.x],usedValues,n,result,inversions,fractionNumber[0], 2);


}
