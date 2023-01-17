/*
    Based on Mark Harris reduction algorithm.
    https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/

#include<stdio.h>
#include<math.h>

__global__ void findMax(int *g_data, int *g_odata, int N){
    __shared__ int sdata[1024];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid]=0;
    if(i < N)
        sdata[tid] = g_data[i];
    // __syncthreads(); // bug

    int s;
    for(s = blockDim.x / 2; s > 0; s /= 2){
        if(tid < s){
            if(sdata[tid] < sdata[tid + s])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
        
    }

    if(tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

//Works for power of 2
__global__ void findMin(int *g_data, int *g_odata, int N){
    __shared__ int sdata[1024];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid]=0;
    if(i < N)
        sdata[tid] = g_data[i];
    __syncthreads();

    int s;
    for(s = blockDim.x / 2; s > 0; s /= 2){
        if(tid < s && i+s < N){ 
            if(sdata[tid] > sdata[tid + s])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
        
    }

    if(tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

int main(int argc, char* argv[]){
    int N = atoi(argv[1]);
    size_t size = N*sizeof(int);
    int limit;

    int *h_in, *h_out;
    h_in = (int*)malloc(size);
    h_out = (int*)malloc(size);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    srand(atoi(argv[1]));
    int i;
    for(i = 0; i < N; i++)
        scanf("%d", &h_in[i]);

    for(i = 0; i < N; i++)
        printf("%d ", h_in[i]);
    printf("\n");

    cudaMemcpy(d_a, h_in, size, cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, size);
    cudaMemset(d_c, 0, size);

    int numBlock = 2;
    int numThread = 1024;

    printf("Blocks: %d Threads: %d\n", numBlock, numThread);
    dim3 gridsize(numBlock,1,1);
    dim3 blocksize(numThread,1,1);

    limit = (int) ceil (((float) N) / blocksize.x);

    findMax<<<gridsize, blocksize>>>(d_a, d_b, N);
    findMax<<<1, blocksize>>>(d_b, d_c, limit);
    cudaMemcpy(h_out, d_c, size, cudaMemcpyDeviceToHost);
    printf("Max %d ", h_out[0]);

    findMin<<<gridsize, blocksize>>>(d_a, d_b, N);
    findMin<<<1, blocksize>>>(d_b, d_c, limit);
    cudaMemcpy(h_out, d_c, size, cudaMemcpyDeviceToHost);

    printf("Min %d ", h_out[0]);
    printf("\n");
    return 0;
}