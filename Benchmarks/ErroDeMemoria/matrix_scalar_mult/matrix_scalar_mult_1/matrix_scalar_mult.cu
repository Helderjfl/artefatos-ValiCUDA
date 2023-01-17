#include <stdio.h>

__global__ void multiMatrix (int *d_a, int scalar, int *d_c, int N) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;                
	int idy = threadIdx.y + blockDim.y * blockIdx.y;                
    int pos = idx + idy * N;                                                                                
	    
    if(idx < N && idy < N)  {
		d_c[pos] = d_a[pos] * scalar;
	} 
} 

int main (int argc, char* argv[]){
	int N = atoi(argv[1]); 
	size_t size = N*N*sizeof(int);
	int num_thread, num_block;

	int *h_a, scalar, *h_c;
	h_a = (int*)malloc(size);
	// scalar = atoi(argv[2]); // bug
	h_c = (int*)malloc(size);

	int *d_a, *d_c;
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_c, size);

    int i = 0;
	for (i = 0; i < N*N; i++){
		h_a[i] = i;
	}
	
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
	cudaMemset(d_c,0,size);

	num_block = 4;
    num_thread = 16;

	printf("Blocks: %d    Threads: %d  \n", num_block*num_block, num_thread*num_thread);

	dim3 gridsize(num_block,num_block,1);
	dim3 blocksize(num_thread,num_thread,1);
  
    multiMatrix<<<gridsize,blocksize>>>(d_a, scalar, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (i = 0; i < N; i++){
		int j;
        for(j = 0; j < N; j++)
		    printf("%d ", h_c[i * N + j]);
        printf("\n");
	}

	cudaFree(d_a);
    cudaFree(d_c);
    free(h_a);
    free(h_c);
}