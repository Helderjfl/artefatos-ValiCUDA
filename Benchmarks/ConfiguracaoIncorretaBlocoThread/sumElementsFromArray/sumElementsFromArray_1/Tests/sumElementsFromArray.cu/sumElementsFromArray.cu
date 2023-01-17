#include <stdio.h>
__device__ void registerTrace(unsigned char *traceArray, int width, int id, int cont, unsigned char node)
{
	traceArray[cont * width + id] = node;
}

unsigned char history[2] = {0, 0};
__host__ void registerTraceHost(FILE *fp, unsigned char node)
{
	if (history[0] != node || history[1] != node)
		fprintf(fp, "%d-0\t", node);
	history[0] = history[1];
	history[1] = node;
}

__global__ void reduceNeighbored(int *d_idata, int *d_odata, unsigned int size, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   unsigned int tid = threadIdx.x;
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
   /* node 2 */
   if(idx >= size) { registerTrace(traceArray, numberOfThreads, map, cont++, 3);  return; }
   registerTrace(traceArray, numberOfThreads, map, cont++, 4);
   int *idata = d_idata + blockIdx.x * blockDim.x;
   /* node 5 */
   if(idata[tid] < 0) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 6);
      idata[tid] = 0;
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 7);
   /* node 8 sync */
   __syncthreads();
   for(/* node 9 */ int stride = 1; /* node 10 */ stride < blockDim.x; /* node 15 */ stride *= 2) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 11);
      if((tid % (2 * stride)) == 0) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 12);
         idata[tid] += idata[tid + stride];
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 13);
      /* node 14 sync */
      __syncthreads();
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 16);
   if(tid == 0) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 17);
      d_odata[blockIdx.x] = idata[0];
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 18);
}

// Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess(int *d_idata, int *d_odata, unsigned int size, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   unsigned int tid = threadIdx.x;
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
   /* node 2 */
   if(idx >= size) { registerTrace(traceArray, numberOfThreads, map, cont++, 3);  return; }
   registerTrace(traceArray, numberOfThreads, map, cont++, 4);
   int *idata = d_idata + blockIdx.x * blockDim.x;
   /* node 5 */
   if(idata[tid] < 0) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 6);
      idata[tid] = 0;
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 7);
   /* node 8 sync */
   __syncthreads();
   for(/* node 9 */ int stride = 1; /* node 10 */ stride < blockDim.x; /* node 16 */ stride *= 2) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 11);
      int index = 2 * stride * tid;
      /* node 12 */
      if(index < blockDim.x) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 13);
         idata[index] += idata[index + stride];
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 14);
      /* node 15 sync */
      __syncthreads();
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 17);
   if(tid == 0) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 18);
      d_odata[blockIdx.x] = idata[0];
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 19);
}

int main(int argc, char **argv) {
   FILE *fp;
   FILE *fpHost = fopen("trace.Host.p0", "w");
   int size = atoi(argv[1]);
   registerTraceHost(fpHost, 1);
   dim3 blocksize(256, 1);
   dim3 gridsize(4, 1);
   size_t bytes = size * sizeof(int);
   int *h_idata = (int *) malloc(bytes);
   int *h_odata = (int *) malloc(gridsize.x * sizeof(int));
   int i;
   for(/* node 2 */ i = 0; /* node 3 */ i < size; /* node 5 */ i++) {
      scanf("%d", &h_idata[i]);
      registerTraceHost(fpHost, 4);
   }
   int gpu_sum;
   int *d_idata = 0;
   int *d_odata = 0;
   registerTraceHost(fpHost, 6);
   cudaMalloc((void **) &d_idata, bytes);
   /* node 7 */
   cudaMalloc((void **) &d_odata, gridsize.x * sizeof(int));
   /* node 8 */
   cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
   /* node 9 */
   cudaDeviceSynchronize();
   int numberOfThreads0 = blocksize.x * blocksize.y * blocksize.z * gridsize.x * gridsize.y * gridsize.z;
   unsigned char *instTrace0;
   cudaMallocManaged(&instTrace0, numberOfThreads0 * 2000 * sizeof(unsigned char));
   /* node 10 */
   reduceNeighbored<<<blocksize, gridsize>>>(d_idata, d_odata, size, instTrace0, numberOfThreads0); // bug
   /* node 11 */
   cudaDeviceSynchronize();
   /* node 12 */
   cudaMemcpy(h_odata, d_odata, gridsize.x * sizeof(int), cudaMemcpyDeviceToHost);
   gpu_sum = 0;
   for(/* node 13 */ i = 0; /* node 14 */ i < gridsize.x; /* node 16 */ i++) {
      gpu_sum += h_odata[i];
      registerTraceHost(fpHost, 15);
   }
   cudaMemset(d_odata, 0, gridsize.x * sizeof(int));
   printf("gpu Neighbored  gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, gridsize.x, blocksize.x);
   registerTraceHost(fpHost, 17);
   cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
   /* node 18 */
   cudaDeviceSynchronize();
   int numberOfThreads1 = gridsize.x * gridsize.y * gridsize.z * blocksize.x * blocksize.y * blocksize.z;
   unsigned char *instTrace1;
   cudaMallocManaged(&instTrace1, numberOfThreads1 * 2000 * sizeof(unsigned char));
   /* node 19 */
   reduceNeighboredLess<<<gridsize, blocksize>>>(d_idata, d_odata, size, instTrace1, numberOfThreads1);
   /* node 20 */
   cudaDeviceSynchronize();
   /* node 21 */
   cudaMemcpy(h_odata, d_odata, gridsize.x * sizeof(int), cudaMemcpyDeviceToHost);
   gpu_sum = 0;
   for(/* node 22 */ i = 0; /* node 23 */ i < gridsize.x; /* node 25 */ i++) {
      gpu_sum += h_odata[i];
      registerTraceHost(fpHost, 24);
   }
   printf("gpu Neighbored2 gpu_sum: %d <<<grid %d block %d>>>\n", gpu_sum, gridsize.x, blocksize.x);
   free(h_idata);
   free(h_odata);
   registerTraceHost(fpHost, 26);
   cudaFree(d_idata);
   registerTraceHost(fpHost, 27);
   cudaFree(d_odata);
   fclose(fpHost);
   fp = fopen("trace.Grid.p2", "w");
   for(int threadIndex = 0; threadIndex < numberOfThreads1; threadIndex++)
   {
   	for(int nodeTrace = 0; nodeTrace < 2000; nodeTrace++)
   	{
   		if(instTrace1[threadIndex + nodeTrace * numberOfThreads1] != 0){
   			if(nodeTrace >= 2){
   				if(instTrace1[threadIndex + nodeTrace * numberOfThreads1] != instTrace1[threadIndex + (nodeTrace-1) * numberOfThreads1] ||
   				instTrace1[threadIndex + nodeTrace * numberOfThreads1] != instTrace1[threadIndex + (nodeTrace-2) * numberOfThreads1])
   					fprintf(fp, "%d-2\t", instTrace1[threadIndex + nodeTrace * numberOfThreads1]);
   			}else
   				fprintf(fp, "%d-2\t", instTrace1[threadIndex + nodeTrace * numberOfThreads1]);
   		}
   	}
   	fprintf(fp, "\n");
   }
   fclose(fp);
   fp = fopen("trace.Grid.p1", "w");
   for(int threadIndex = 0; threadIndex < numberOfThreads0; threadIndex++)
   {
   	for(int nodeTrace = 0; nodeTrace < 2000; nodeTrace++)
   	{
   		if(instTrace0[threadIndex + nodeTrace * numberOfThreads0] != 0){
   			if(nodeTrace >= 2){
   				if(instTrace0[threadIndex + nodeTrace * numberOfThreads0] != instTrace0[threadIndex + (nodeTrace-1) * numberOfThreads0] ||
   				instTrace0[threadIndex + nodeTrace * numberOfThreads0] != instTrace0[threadIndex + (nodeTrace-2) * numberOfThreads0])
   					fprintf(fp, "%d-1\t", instTrace0[threadIndex + nodeTrace * numberOfThreads0]);
   			}else
   				fprintf(fp, "%d-1\t", instTrace0[threadIndex + nodeTrace * numberOfThreads0]);
   		}
   	}
   	fprintf(fp, "\n");
   }
   fclose(fp);
   fp = fopen("commsize", "w");
   fprintf(fp, "3");
   fclose(fp);
   
   return 0;
}
