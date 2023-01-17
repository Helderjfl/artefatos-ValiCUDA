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

/*
Based on Mark Harris reduction algorithm.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/
__global__ void findMax(int *g_data, int *g_odata, int N, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   int sdata[1024]; // bug
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   sdata[tid] = 0;
   /* node 2 */
   if(i < N) { registerTrace(traceArray, numberOfThreads, map, cont++, 3); sdata[tid] = g_data[i]; }
   registerTrace(traceArray, numberOfThreads, map, cont++, 4);
   /* node 5 sync */
   __syncthreads();
   int s;
   for(/* node 6 */ s = blockDim.x / 2; /* node 7 */ s > 0; /* node 14 */ s /= 2) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 8);
      if(tid < s) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 9);
         if(sdata[tid] < sdata[tid + s]) { registerTrace(traceArray, numberOfThreads, map, cont++, 10); sdata[tid] = sdata[tid + s]; }
         registerTrace(traceArray, numberOfThreads, map, cont++, 11);
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 12);
      /* node 13 sync */
      __syncthreads();
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 15);
   if(tid == 0) { registerTrace(traceArray, numberOfThreads, map, cont++, 16); g_odata[blockIdx.x] = sdata[0]; }
   registerTrace(traceArray, numberOfThreads, map, cont++, 17);
}

//Works for power of 2
__global__ void findMin(int *g_data, int *g_odata, int N, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   static __shared__ int sdata[1024];
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   sdata[tid] = 0;
   /* node 2 */
   if(i < N) { registerTrace(traceArray, numberOfThreads, map, cont++, 3); sdata[tid] = g_data[i]; }
   registerTrace(traceArray, numberOfThreads, map, cont++, 4);
   /* node 5 sync */
   __syncthreads();
   int s;
   for(/* node 6 */ s = blockDim.x / 2; /* node 7 */ s > 0; /* node 14 */ s /= 2) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 8);
      if(tid < s && i + s < N) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 9);
         if(sdata[tid] > sdata[tid + s]) { registerTrace(traceArray, numberOfThreads, map, cont++, 10); sdata[tid] = sdata[tid + s]; }
         registerTrace(traceArray, numberOfThreads, map, cont++, 11);
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 12);
      /* node 13 sync */
      __syncthreads();
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 15);
   if(tid == 0) { registerTrace(traceArray, numberOfThreads, map, cont++, 16); g_odata[blockIdx.x] = sdata[0]; }
   registerTrace(traceArray, numberOfThreads, map, cont++, 17);
}

int main(int argc, char *argv[]) {
   FILE *fp;
   FILE *fpHost = fopen("trace.Host.p0", "w");
   int N = atoi(argv[1]);
   registerTraceHost(fpHost, 1);
   size_t size = N * sizeof(int);
   int limit;
   int *h_in, *h_out;
   h_in = (int *) malloc(size);
   h_out = (int *) malloc(size);
   int *d_a, *d_b, *d_c;
   /* node 2 */
   cudaMalloc(&d_a, size);
   /* node 3 */
   cudaMalloc(&d_b, size);
   /* node 4 */
   cudaMalloc(&d_c, size);
   srand(atoi(argv[1]));
   int i;
   for(/* node 5 */ i = 0; /* node 6 */ i < N; /* node 8 */ i++) {
      scanf("%d", &h_in[i]);
      registerTraceHost(fpHost, 7);
   }
   for(registerTraceHost(fpHost, 9), i = 0; /* node 10 */ i < N; /* node 12 */ i++) {
      printf("%d ", h_in[i]);
      registerTraceHost(fpHost, 11);
   }
   printf("\n");
   registerTraceHost(fpHost, 13);
   cudaMemcpy(d_a, h_in, size, cudaMemcpyHostToDevice);
   cudaMemset(d_b, 0, size);
   cudaMemset(d_c, 0, size);
   int numBlock = 4;
   int numThread = 512;
   printf("Blocks: %d Threads: %d\n", numBlock, numThread);
   dim3 gridsize(numBlock, 1, 1);
   dim3 blocksize(numThread, 1, 1);
   limit = (int) ceil(((float) N) / blocksize.x);
   int numberOfThreads0 = gridsize.x * gridsize.y * gridsize.z * blocksize.x * blocksize.y * blocksize.z;
   unsigned char *instTrace0;
   cudaMallocManaged(&instTrace0, numberOfThreads0 * 2000 * sizeof(unsigned char));
   /* node 14 */
   findMax<<<gridsize, blocksize>>>(d_a, d_b, N, instTrace0, numberOfThreads0);
   int numberOfThreads1 = 1 * blocksize;
   unsigned char *instTrace1;
   cudaMallocManaged(&instTrace1, numberOfThreads1 * 2000 * sizeof(unsigned char));
   /* node 15 */
   findMax<<<1, blocksize>>>(d_b, d_c, limit, instTrace1, numberOfThreads1);
   /* node 16 */
   cudaMemcpy(h_out, d_c, size, cudaMemcpyDeviceToHost);
   printf("Max %d ", h_out[0]);
   int numberOfThreads2 = gridsize.x * gridsize.y * gridsize.z * blocksize.x * blocksize.y * blocksize.z;
   unsigned char *instTrace2;
   cudaMallocManaged(&instTrace2, numberOfThreads2 * 2000 * sizeof(unsigned char));
   /* node 17 */
   findMin<<<gridsize, blocksize>>>(d_a, d_b, N, instTrace2, numberOfThreads2);
   int numberOfThreads3 = 1 * blocksize;
   unsigned char *instTrace3;
   cudaMallocManaged(&instTrace3, numberOfThreads3 * 2000 * sizeof(unsigned char));
   /* node 18 */
   findMin<<<1, blocksize>>>(d_b, d_c, limit, instTrace3, numberOfThreads3);
   registerTraceHost(fpHost, 19);
   cudaMemcpy(h_out, d_c, size, cudaMemcpyDeviceToHost);
   printf("Min %d ", h_out[0]);
   printf("\n");
   fclose(fpHost);
   fp = fopen("trace.Grid.p4", "w");
   for(int threadIndex = 0; threadIndex < numberOfThreads3; threadIndex++)
   {
   	for(int nodeTrace = 0; nodeTrace < 2000; nodeTrace++)
   	{
   		if(instTrace3[threadIndex + nodeTrace * numberOfThreads3] != 0){
   			if(nodeTrace >= 2){
   				if(instTrace3[threadIndex + nodeTrace * numberOfThreads3] != instTrace3[threadIndex + (nodeTrace-1) * numberOfThreads3] ||
   				instTrace3[threadIndex + nodeTrace * numberOfThreads3] != instTrace3[threadIndex + (nodeTrace-2) * numberOfThreads3])
   					fprintf(fp, "%d-4\t", instTrace3[threadIndex + nodeTrace * numberOfThreads3]);
   			}else
   				fprintf(fp, "%d-4\t", instTrace3[threadIndex + nodeTrace * numberOfThreads3]);
   		}
   	}
   	fprintf(fp, "\n");
   }
   fclose(fp);
   fp = fopen("trace.Grid.p3", "w");
   for(int threadIndex = 0; threadIndex < numberOfThreads2; threadIndex++)
   {
   	for(int nodeTrace = 0; nodeTrace < 2000; nodeTrace++)
   	{
   		if(instTrace2[threadIndex + nodeTrace * numberOfThreads2] != 0){
   			if(nodeTrace >= 2){
   				if(instTrace2[threadIndex + nodeTrace * numberOfThreads2] != instTrace2[threadIndex + (nodeTrace-1) * numberOfThreads2] ||
   				instTrace2[threadIndex + nodeTrace * numberOfThreads2] != instTrace2[threadIndex + (nodeTrace-2) * numberOfThreads2])
   					fprintf(fp, "%d-3\t", instTrace2[threadIndex + nodeTrace * numberOfThreads2]);
   			}else
   				fprintf(fp, "%d-3\t", instTrace2[threadIndex + nodeTrace * numberOfThreads2]);
   		}
   	}
   	fprintf(fp, "\n");
   }
   fclose(fp);
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
   fprintf(fp, "5");
   fclose(fp);
   
   return 0;
}
