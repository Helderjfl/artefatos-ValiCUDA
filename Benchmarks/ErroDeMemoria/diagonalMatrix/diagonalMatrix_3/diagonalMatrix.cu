#include <stdio.h>

__global__ void diagonal (int *d_a, int *d_b, int N, int mode) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;                
	int idy = threadIdx.y + blockDim.y * blockIdx.y;                
    int pos = idx + idy * N;                                                                         

    if(mode == 0){ // main diagonal
		if(idx < N && idy < N)  {
			if(idx == idy)
				d_b[pos] = d_a[pos];
			else
				d_b[pos] = 0;
		} 
	}
	if(mode == 1){ // secondary diagonal
		if(idx < N && idy < N)  {
			if(idx + idy == N - 1)
				d_b[pos] = d_a[pos];
			else
				d_b[pos] = 0;
		} 
	}
	if(mode == 2){ // inverse main diagonal
		int posb = N * N -1 - pos + 1; // bug
		if(idx < N && idy < N)  {
			if(idx == idy)
				d_b[pos] = d_a[posb];
			else
				d_b[pos] = 0;
		}
	}
	if(mode == 3){ // inverse secondary diagonal
		int posb = idx * N + idy;
		if(idx < N && idy < N)  {
			if(idx + idy == N - 1)
				d_b[pos] = d_a[posb];
			else
				d_b[pos] = 0;
		}
	}
} 

int main (int argc, char* argv[]){
	int N = atoi(argv[1]);
	int mode = atoi(argv[2]);
	size_t size = N*N*sizeof(int);
	int num_thread, num_block;
    int i, j;

	int *h_a, *h_b;
	h_a = (int*)malloc(size);
	h_b = (int*)malloc(size);

	int *d_a, *d_b;
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);

	for (i = 0; i < N*N; i++){
		scanf("%d", &h_a[i]);
	}
	
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, size);

	num_block = 2;
    num_thread = 32;

	printf("Blocks: %d    Threads: %d  \n", num_block, num_thread);

	dim3 gridsize(num_block,num_block,1);
	dim3 blocksize(num_thread,num_thread,1);
  
    diagonal<<<gridsize,blocksize>>>(d_a, d_b, N, mode);

    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    for (i = 0; i < N; i++){
        for(j = 0; j < N; j++)
		    printf("%d ", h_b[i * N + j]);
        printf("\n");
	}

	cudaFree(d_a);
	cudaFree(d_b);
    free(h_a);
    free(h_b);
}