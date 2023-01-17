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

__global__ void multiply(long *d_a, long *d_b, long *d_c, long N, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   long idx = threadIdx.x + blockDim.x * blockIdx.x;
   long idy = threadIdx.y + blockDim.y * blockIdx.y;
   long pos = idx + idy * N;
   long tempValue = 0, k;
   extern __shared__ long array[];
   /* node 2 */
   if(idx < N) { registerTrace(traceArray, numberOfThreads, map, cont++, 3); array[idx] = d_b[idx]; }
   registerTrace(traceArray, numberOfThreads, map, cont++, 4);
   // __syncthreads(); // bug
   /* node 5 */
   if(idx < N) {
      for(registerTrace(traceArray, numberOfThreads, map, cont++, 6), k = 0; /* node 7 */ k < N; /* node 9 */ k++) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 8);
         tempValue += d_a[idx * N + k] * array[k];
      }
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 10);
   /* node 11 */
   if(pos < N) { registerTrace(traceArray, numberOfThreads, map, cont++, 12); d_c[idx] = tempValue; }
   registerTrace(traceArray, numberOfThreads, map, cont++, 13);
}

int main(int argc, char *argv[]) {
   FILE *fp;
   FILE *fpHost = fopen("trace.Host.p0", "w");
   int N = atoi(argv[1]);
   registerTraceHost(fpHost, 1);
   size_t size = N * N * sizeof(long);
   int num_thread, num_block;
   int sharedMemory = atoi(argv[2]);
   long *h_a, *h_b, *h_c;
   h_a = (long *) malloc(size);
   h_b = (long *) malloc(N * sizeof(long));
   h_c = (long *) malloc(N * sizeof(long));
   long *d_a, *d_b, *d_c;
   /* node 2 */
   cudaMalloc(&d_a, size);
   /* node 3 */
   cudaMalloc(&d_b, N * sizeof(long));
   /* node 4 */
   cudaMalloc(&d_c, N * sizeof(long));
   int i, j;
   for(/* node 5 */ i = 0; /* node 6 */ i < N; registerTraceHost(fpHost, 11), i++) {
      for(registerTraceHost(fpHost, 7), j = 0; /* node 8 */ j < N; /* node 10 */ j++) {
         scanf("%ld", &h_a[i * N + j]);
         registerTraceHost(fpHost, 9);
      }
   }
   for(registerTraceHost(fpHost, 12), i = 0; /* node 13 */ i < N; /* node 15 */ i++) {
      scanf("%ld", &h_b[i]);
      registerTraceHost(fpHost, 14);
   }
   registerTraceHost(fpHost, 16);
   cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
   /* node 17 */
   cudaMemcpy(d_b, h_b, N * sizeof(long), cudaMemcpyHostToDevice);
   cudaMemset(d_c, 0, N * sizeof(long));
   num_block = 1;
   num_thread = 1024;
   printf("Blocks: %d    Threads: %d  \n", num_block, num_thread);
   dim3 gridsize(num_block, 1, 1);
   dim3 blocksize(num_thread, 1, 1);
   int numberOfThreads0 = gridsize.x * gridsize.y * gridsize.z * blocksize.x * blocksize.y * blocksize.z;
   unsigned char *instTrace0;
   cudaMallocManaged(&instTrace0, numberOfThreads0 * 2000 * sizeof(unsigned char));
   /* node 18 */
   multiply<<<gridsize, blocksize, sharedMemory * sizeof(long)>>>(d_a, d_b, d_c, N, instTrace0, numberOfThreads0);
   /* node 19 */
   cudaMemcpy(h_c, d_c, N * sizeof(long), cudaMemcpyDeviceToHost);
   for(/* node 20 */ i = 0; /* node 21 */ i < N; /* node 23 */ i++) {
      printf("%ld ", h_c[i]);
      registerTraceHost(fpHost, 22);
   }
   printf("\n");
   registerTraceHost(fpHost, 24);
   cudaFree(d_a);
   /* node 25 */
   cudaFree(d_b);
   registerTraceHost(fpHost, 26);
   cudaFree(d_c);
   free(h_a);
   free(h_b);
   free(h_c);
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
}
