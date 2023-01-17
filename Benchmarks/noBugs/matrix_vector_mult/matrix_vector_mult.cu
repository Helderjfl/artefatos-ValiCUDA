#include <stdio.h>

__global__ void multiply (long int *d_a, long int *d_b, long int *d_c, long int N) {
	long int idx = threadIdx.x + blockDim.x * blockIdx.x;                
	long int idy = threadIdx.y + blockDim.y * blockIdx.y;                
    long int pos = idx + idy * N;                                                                                
	long int tempValue = 0, k;
	    
	extern __shared__ long int array[];

	if(idx < N)
		array[idx] = d_b[idx];

	__syncthreads();

    if(idx < N && idy < N)  {
		for(k = 0; k < N; k++){
			tempValue += d_a[idx*N+k] * array[k];
		}
	} 

	if(pos < N)
		d_c[idx] = tempValue;
} 

int main (int argc, char* argv[]){
	int N = atoi(argv[1]); 
	size_t size = N*N*sizeof(long int);
	int num_thread, num_block;
	int sharedMemory = atoi(argv[2]); 

	long int *h_a, *h_b, *h_c;
	h_a = (long int*)malloc(size);
	h_b = (long int*)malloc(N*sizeof(long int));
	h_c = (long int*)malloc(N*sizeof(long int));

	long int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, N*sizeof(long int));
	cudaMalloc(&d_c, N*sizeof(long int));

    int i, j;
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			scanf("%ld", &h_a[i * N + j]);
		}
	}

    for(i = 0; i < N; i++){
		scanf("%ld", &h_b[i]);
	}
	
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(long int), cudaMemcpyHostToDevice);
	cudaMemset(d_c,0,N*sizeof(long int));

	num_block = 1;
    num_thread = 1024;

	printf("Blocks: %d    Threads: %d  \n", num_block, num_thread);

	dim3 gridsize(num_block,1,1);
	dim3 blocksize(num_thread,1,1);
  
    multiply<<<gridsize,blocksize, sharedMemory * sizeof(long int)>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N*sizeof(long int), cudaMemcpyDeviceToHost);

    for (i = 0; i < N; i++){
		printf("%ld ", h_c[i]);
	}
	printf("\n");

	cudaFree(d_a);
	cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
	free(h_b);
    free(h_c);
}