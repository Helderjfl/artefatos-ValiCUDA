#include <stdio.h>
#include <math.h>
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

__global__ void count(int *g_data, int *g_odata, int value, int N, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   int sdata[1024]; // bug
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   sdata[tid] = 0;
   /* node 2 */
   if(i < N) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 3);
      if(g_data[i] == value) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 4);
         sdata[tid] = 1;
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 5);
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 6);
   /* node 7 sync */
   __syncthreads();
   int s;
   for(/* node 8 */ s = blockDim.x / 2; registerTrace(traceArray, numberOfThreads, map, cont++, 9), s > 0; /* node 14 */ s /= 2) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 10);
      if(tid < s) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 11);
         sdata[tid] += sdata[tid + s];
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 12);
      /* node 13 sync */
      __syncthreads();
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 15);
   if(tid == 0) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 16);
      g_odata[blockIdx.x] = sdata[0];
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 17);
}

int main(int argc, char *argv[]) {
   FILE *fp;
   FILE *fpHost = fopen("trace.Host.p0", "w");
   int numBlock = 8;
   registerTraceHost(fpHost, 1);
   int numThread = 1024;
   printf("Blocks: %d Threads: %d\n", numBlock, numThread);
   dim3 gridsize(numBlock, 1, 1);
   dim3 blocksize(numThread, 1, 1);
   int N = atoi(argv[1]);
   int value = atoi(argv[2]);
   size_t size = N * sizeof(int);
   size_t sizeOutput = numBlock * sizeof(int);
   int *h_in, *h_out;
   h_in = (int *) malloc(size);
   h_out = (int *) malloc(sizeOutput);
   int *g_data, *g_odata;
   /* node 2 */
   cudaMalloc(&g_data, size);
   /* node 3 */
   cudaMalloc(&g_odata, sizeOutput);
   int i;
   for(/* node 4 */ i = 0; /* node 5 */ i < N; /* node 7 */ i++) {
      scanf("%d", &h_in[i]);
      registerTraceHost(fpHost, 6);
   }
   registerTraceHost(fpHost, 8);
   cudaMemcpy(g_data, h_in, size, cudaMemcpyHostToDevice);
   cudaMemset(g_odata, 0, sizeOutput);
   int numberOfThreads0 = gridsize.x * gridsize.y * gridsize.z * blocksize.x * blocksize.y * blocksize.z;
   unsigned char *instTrace0;
   cudaMallocManaged(&instTrace0, numberOfThreads0 * 2000 * sizeof(unsigned char));
   /* node 9 */
   count<<<gridsize, blocksize>>>(g_data, g_odata, value, N, instTrace0, numberOfThreads0);
   /* node 10 */
   cudaMemcpy(h_out, g_odata, sizeOutput, cudaMemcpyDeviceToHost);
   /* node 11 */
   if(N > 1024) {
      for(registerTraceHost(fpHost, 12), i = 1; /* node 13 */ i < numBlock; /* node 15 */ i++) {
         h_out[0] += h_out[i];
         registerTraceHost(fpHost, 14);
      }
   }
   registerTraceHost(fpHost, 16);
   printf("Ocurrences: %d\n", h_out[0]);
   fclose(fpHost);
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
   fprintf(fp, "2");
   fclose(fp);
   
   return 0;
}
