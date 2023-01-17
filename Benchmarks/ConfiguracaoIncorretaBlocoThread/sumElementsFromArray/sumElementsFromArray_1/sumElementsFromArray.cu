#include <stdio.h>

__global__ void reduceNeighbored (int *d_idata, int *d_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;
    
    int *idata = d_idata + blockIdx.x * blockDim.x;

    if (idata[tid] < 0)
    {
        idata[tid] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
         d_odata[blockIdx.x] = idata[0];
    }
}

// Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess (int *d_idata, int *d_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= size) return;

    int *idata = d_idata + blockIdx.x * blockDim.x;

    if (idata[tid] < 0)
    {
        idata[tid] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;

        if (index < blockDim.x)
        {

            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
         d_odata[blockIdx.x] = idata[0];
    }
}

int main(int argc, char **argv){
    int size = atoi(argv[1]); 
    dim3 blocksize (256, 1);
    dim3 gridsize  (4, 1);

    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(gridsize.x * sizeof(int));

    int i;
    for (i = 0; i < size; i++)
    {
        scanf("%d", &h_idata[i]);
    }

    int gpu_sum;
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, gridsize.x * sizeof(int));

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reduceNeighbored<<<blocksize, gridsize>>>(d_idata, d_odata, size);  // bug
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, gridsize.x * sizeof(int), cudaMemcpyDeviceToHost);
    
    gpu_sum = 0;
    for (i = 0; i < gridsize.x; i++) 
        gpu_sum += h_odata[i];

    cudaMemset(d_odata, 0, gridsize.x * sizeof(int));
    printf("gpu Neighbored  gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, gridsize.x, blocksize.x);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reduceNeighboredLess<<<gridsize, blocksize>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, gridsize.x * sizeof(int), cudaMemcpyDeviceToHost);
    
    gpu_sum = 0;
    for (i = 0; i < gridsize.x; i++) 
        gpu_sum += h_odata[i];

    printf("gpu Neighbored2 gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, gridsize.x, blocksize.x);


    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}