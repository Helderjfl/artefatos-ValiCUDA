#include<stdio.h>
#include<math.h>

__global__ void count(int *g_data, int *g_odata, int value, int N){
    __shared__ int sdata[1024];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;
    if(i < N){
        if(g_data[i] == value){
            sdata[tid] = 1;
        }
    }
    // __syncthreads(); // bug

    int s;
    for(s = blockDim.x / 2; s > 0; s /= 2){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if(tid == 0){
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char* argv[]){
    int numBlock = 8;
    int numThread = 1024;

    printf("Blocks: %d Threads: %d\n", numBlock, numThread);
    dim3 gridsize(numBlock, 1, 1);
    dim3 blocksize(numThread, 1, 1);

    int N = atoi(argv[1]);
    int value = atoi(argv[2]);
    size_t size = N * sizeof(int);
    size_t sizeOutput = numBlock * sizeof(int);

    int *h_in, *h_out;
    h_in = (int*) malloc(size);
    h_out = (int*) malloc(sizeOutput);

    int *g_data, *g_odata;
    cudaMalloc(&g_data, size);
    cudaMalloc(&g_odata, sizeOutput);

    int i;
    for(i = 0; i < N; i++){
        scanf("%d", &h_in[i]);
    }

    cudaMemcpy(g_data, h_in, size, cudaMemcpyHostToDevice);
    cudaMemset(g_odata, 0, sizeOutput);

    count<<<gridsize, blocksize>>>(g_data, g_odata, value, N);
    cudaMemcpy(h_out, g_odata, sizeOutput, cudaMemcpyDeviceToHost);

    if(N > 1024){
        for(i = 1; i < numBlock; i++){
            h_out[0] += h_out[i];
        }
    }

    printf("Ocurrences: %d\n", h_out[0]);

    return 0;
}