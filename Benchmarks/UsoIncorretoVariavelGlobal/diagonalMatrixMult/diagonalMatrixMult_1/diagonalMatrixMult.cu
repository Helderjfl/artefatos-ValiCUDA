#include <stdio.h>

__global__ void diagonalMult (int *d_a, int *d_b, int number, int N, int mode) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;                
	int idy = threadIdx.y + blockDim.y * blockIdx.y;                
    int pos = idx + idy * N;  
	int posb, conda, condb;     
	
	__shared__ int s_a[32][32];
	
	if(mode == 0){
		posb = pos;
		conda = idx;
		condb = idy;
	}
	if (mode == 1){
		posb = pos;
		conda = idx + idy;
		condb = N - 1;
	}
	if(mode == 2){
		posb = N * N -1 - pos;
		conda = idx;
		condb = idy;
	}
	if(mode == 3){
		posb = N * N -1 - pos;
		conda = idx + idy;
		condb = N - 1;
	}

	if(idx < N && idy < N)
		s_a[threadIdx.x][threadIdx.y] = d_a[posb];
    
    if(idx < N && idy < N)  {
        if(conda == condb){
		    d_b[pos] = s_a[threadIdx.x][threadIdx.y] * number;
		}
	} 
} 

int main (int argc, char* argv[]){
	int N = atoi(argv[1]); 
	size_t size = N*N*sizeof(int);
	int num_thread, num_block;
	int mode = atoi(argv[2]);

	int *h_a, *h_b, number;
	h_a = (int*)malloc(size);
	h_b = (int*)malloc(size);
	number = 4;

	int *d_a, *d_b;
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);

    int i = 0;
	for (i = 0; i < N*N; i++){
		scanf("%d", &h_a[i]);
	}
	
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,d_a, size, cudaMemcpyDeviceToDevice);

	num_block = 2;
    num_thread = 32;

	printf("Blocks: %d    Threads: %d  \n", num_block, num_thread);

	dim3 gridsize(num_block,num_block,1);
	dim3 blocksize(num_thread,num_thread,1);
  
    diagonalMult<<<gridsize,blocksize>>>(d_a, d_a, number, N, mode); // bug

    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    for (i = 0; i < N; i++){
		int j;
        for(j = 0; j < N; j++){
		    printf("%d ", h_b[i * N + j]);
		}
        printf("\n");
	}

	cudaFree(d_a);
	cudaFree(d_b);
    free(h_a);
    free(h_b);
}