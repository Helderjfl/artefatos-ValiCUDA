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

__global__ void multiMatrix(int *d_a, int scalar, int *d_c, int N, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   int idx = threadIdx.x + blockDim.x * blockIdx.x;
   int idy = threadIdx.y + blockDim.y * blockIdx.y;
   int pos = idx + idy * N;
   /* node 2 */
   if(idx < N && idy < N) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 3);
      d_c[pos] = d_a[pos] * scalar;
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 4);
}

int main(int argc, char *argv[]) {
   FILE *fp;
   FILE *fpHost = fopen("trace.Host.p0", "w");
   int N = atoi(argv[1]);
   registerTraceHost(fpHost, 1);
   size_t size = N * N * sizeof(int);
   int num_thread, num_block;
   int *h_a;
   int scalar;
   int *h_c;
   h_a = (int *) malloc(size);
   // scalar = atoi(argv[2]); // bug
   h_c = (int *) malloc(size);
   int *d_a, *d_c;
   /* node 2 */
   cudaMalloc(&d_a, size);
   /* node 3 */
   cudaMalloc(&d_c, size);
   int i = 0;
   for(/* node 4 */ i = 0; /* node 5 */ i < N * N; /* node 7 */ i++) {
      h_a[i] = i;
      registerTraceHost(fpHost, 6);
   }
   registerTraceHost(fpHost, 8);
   cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
   cudaMemset(d_c, 0, size);
   num_block = 4;
   num_thread = 16;
   printf("Blocks: %d    Threads: %d  \n", num_block * num_block, num_thread * num_thread);
   dim3 gridsize(num_block, num_block, 1);
   dim3 blocksize(num_thread, num_thread, 1);
   int numberOfThreads0 = gridsize.x * gridsize.y * gridsize.z * blocksize.x * blocksize.y * blocksize.z;
   unsigned char *instTrace0;
   cudaMallocManaged(&instTrace0, numberOfThreads0 * 2000 * sizeof(unsigned char));
   /* node 9 */
   multiMatrix<<<gridsize, blocksize>>>(d_a, scalar, d_c, N, instTrace0, numberOfThreads0);
   /* node 10 */
   cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
   for(/* node 11 */ i = 0; /* node 12 */ i < N; registerTraceHost(fpHost, 18), i++) {
      int j;
      registerTraceHost(fpHost, 13);
      for(/* node 14 */ j = 0; /* node 15 */ j < N; /* node 17 */ j++) {
         printf("%d ", h_c[i * N + j]);
         registerTraceHost(fpHost, 16);
      }
      printf("\n");
   }
   registerTraceHost(fpHost, 19);
   cudaFree(d_a);
   registerTraceHost(fpHost, 20);
   cudaFree(d_c);
   free(h_a);
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
