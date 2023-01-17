#include <stdio.h>

__global__ void vectorAdd(int *d_a, int *d_b, int *d_c){
    int id = threadIdx.x;

    d_c[id] = d_a[id] + d_b[id];
}

int main(int argc, char **argv){
    int N = atoi(argv[1]);
    int i;
    int *a;
    int *b;
    int *c;

    size_t size = N*sizeof(int);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);
    
    for(i = 0; i < N; i++){
        a[i] = b[i] = i;
    }
    
    int *d_a;
    int *d_b;
    int *d_c;


    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    vectorAdd<<<1, 1024>>>(d_a, d_b, d_c);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for(i = 0; i < N; i++)
    printf("%d ", c[i]);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}