#include <stdio.h>
#define TILE_DIM 16

__global__ void sumMatrix (int *d_a, int *d_b, int *d_c, int N) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;                
	int idy = threadIdx.y + blockDim.y * blockIdx.y;                
    int pos = idx + idy * N;                                                                         

	__shared__ int s_a[32][32];
	__shared__ int s_b[32][32];

	if(idx < N && idy < N){
		s_a[threadIdx.y][threadIdx.x] = d_a[pos];
		s_b[threadIdx.y][threadIdx.x] = d_b[pos];
	}
    
    if(idx < N && idy < N)  {
		if(s_a[threadIdx.y][threadIdx.x] < 0){
			s_a[threadIdx.y][threadIdx.x] = s_a[threadIdx.y][threadIdx.x] * -1;
		}
	
		if(s_b[threadIdx.y][threadIdx.x] < 0){
			s_b[threadIdx.y][threadIdx.x] = s_b[threadIdx.y][threadIdx.x] * -1;
		}

		d_c[pos] = s_a[idy][idx] + s_b[threadIdx.y][threadIdx.x]; // bug
	} 
} 

int main (int argc, char* argv[]){
	int N = atoi(argv[1]); 
	size_t size = N*N*sizeof(int);

	int *h_a, *h_b, *h_c;
	h_a = (int*)malloc(size);
	h_b = (int*)malloc(size);
	h_c = (int*)malloc(size);

	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

    int i;
	for (i = 0; i < N*N; i++){
		scanf("%d", &h_a[i]);
	}

	for (i = 0; i < N*N; i++){
		scanf("%d", &h_b[i]);
	}
	
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);
	cudaMemset(d_c,0,size);

	dim3 gridsize(4,4,1);
	dim3 blocksize(32,32,1);
  
    sumMatrix<<<gridsize,blocksize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	for(i = 0; i < N; i++){
		int j;
		for(j = 0; j < N; j++)
			printf("%d ", h_c[i * N + j]);
		printf("\n");
	}

	cudaFree(d_a);
	cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}