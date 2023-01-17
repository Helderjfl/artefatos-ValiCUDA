#include <stdio.h>

__global__ void multiMatrix (int *d_a, int *d_b, int *d_c, int N, int TILE_DIM) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;                
	int idy = threadIdx.y + blockDim.y * blockIdx.y;                
    int pos = idx + idy * N;                                        
	int temp_result = 0;  
	int posa, posb;                                          

	extern __shared__ int s[];
	int *s_a = s;
	int *s_b = &s[TILE_DIM * TILE_DIM];

	int tile_idx;
	for (tile_idx = 0; tile_idx < gridDim.x; tile_idx++) 
	{
        posa = idy * N + (tile_idx * TILE_DIM + threadIdx.x);
        posb = (tile_idx * TILE_DIM + threadIdx.y) * N + idx;
		
		if (posa < N*N) {
			s_a[threadIdx.y * TILE_DIM + threadIdx.x] = d_a[posa];
        } else {
            s_a[threadIdx.y * TILE_DIM + threadIdx.x] = 0;
        }
        if (posb < N*N) {
            s_b[threadIdx.y * TILE_DIM + threadIdx.x] = d_b[posb];
        } else {
            s_b[threadIdx.y * TILE_DIM + threadIdx.x] = 0;
        }
		__syncthreads();

		if (idx < N && idy < N) {
			int i;
			
			for (i = 0; i < TILE_DIM; i++) {
				if(s_a[threadIdx.y * TILE_DIM + i] != 0 && s_b[threadIdx.x + TILE_DIM * i] != 0){
					temp_result += s_a[threadIdx.y * TILE_DIM + i] * s_b[threadIdx.x + TILE_DIM * i]; 
				}
			}
		}
            
        __syncthreads();  
    }
    __syncthreads();   
    
    if(idx < N && idy < N)  {
		d_c[pos] = temp_result;
	} 
} 

int main (int argc, char* argv[]){
	int N = atoi(argv[1]); 
	int TILE_DIM = atoi(argv[2]);
	size_t size = N * N * sizeof(int);
	int num_thread, num_block;

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
	
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	cudaMemset(d_c, 0, size);

	num_block = 4;
    num_thread = 16;

	dim3 gridsize(num_block, num_block, 1);
	dim3 blocksize(num_thread, num_thread, 1);
  
    multiMatrix<<<gridsize,blocksize, 2*TILE_DIM * TILE_DIM * sizeof(int)>>>(d_a, d_b, d_c, N, TILE_DIM);

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