#include <stdio.h>

__global__ void multiply (int *d_a, int *d_b, int *d_c, int N) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;                
	int idy = threadIdx.y + blockDim.y * blockIdx.y;                
    int pos = idx + idy * N;                                                                                
	int tempValue = 0, k;
	    
	extern __shared__ int array[];

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
	size_t size = N*N*sizeof(int);
	int num_thread, num_block;
	int sharedMemory = atoi(argv[2]); 

	int *h_a, *h_b, *h_c;
	h_a = (int*)malloc(size);
	h_b = (int*)malloc(N*sizeof(int));
	h_c = (int*)malloc(N*sizeof(int));

	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, N*sizeof(int));
	cudaMalloc(&d_c, N*sizeof(int));

    int i, j;
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			scanf("%d", &h_a[i * N + j]);
		}
	}

    for(i = 0; i < N; i++){
		scanf("%d", &h_b[i]);
	}
	
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N, cudaMemcpyHostToDevice); // bug
	cudaMemset(d_c,0,N*sizeof(int));

	num_block = 1;
    num_thread = 256;

	printf("Blocks: %d    Threads: %d  \n", num_block, num_thread);

	dim3 gridsize(num_block,1,1);
	dim3 blocksize(num_thread,1,1);
  
    multiply<<<gridsize,blocksize, sharedMemory * sizeof(int)>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (i = 0; i < N; i++){
		printf("%d ", h_c[i]);
	}
	printf("\n");

	cudaFree(d_a);
	cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
	free(h_b);
    free(h_c);
}