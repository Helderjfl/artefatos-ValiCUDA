#include<stdio.h>

__global__ void verifyArray(int *d_a, int *d_b, bool *d_equal, int N){
    int pos = threadIdx.x + blockDim.x * blockIdx.x;

    if(d_a[pos] != d_b[pos] && pos < N)
        *d_equal = false;
}

int main(int argc, char* argv[]){
    int N = atoi(argv[1]);
    size_t size = N*sizeof(int);
    bool *h_equal = (bool*)malloc(sizeof(bool));
    *h_equal = true;

    int *h_a, *h_b;
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);

    int *d_a, *d_b;
    bool *d_equal;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_equal, sizeof(bool));

    int i;
	for (i = 0; i < N; i++){
        scanf("%d", &h_a[i]);
	}

    for (i = 0; i < N; i++){
        scanf("%d", &h_b[i]);
	}

    int num_block = 32;
    int num_thread = 256;

    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_a,h_b,size,cudaMemcpyHostToDevice); // bug

    dim3 gridsize(num_block, 1, 1);
    dim3 blocksize(num_thread, 1, 1);

    cudaMemcpy(d_equal, h_equal, sizeof(bool), cudaMemcpyHostToDevice);
    verifyArray<<<gridsize, blocksize>>>(d_a, d_b, d_equal, N);
    cudaMemcpy(h_equal, d_equal, sizeof(bool), cudaMemcpyDeviceToHost);
    printf("%s\n", * h_equal ? "true" : "false");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_equal);
    free(h_a);
    free(h_b);
    free(h_equal);
}