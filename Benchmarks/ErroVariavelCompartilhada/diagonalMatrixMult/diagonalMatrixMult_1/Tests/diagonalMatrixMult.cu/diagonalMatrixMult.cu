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

__global__ void diagonalMult(int *d_a, int *d_b, int number, int N, int mode, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   int idx = threadIdx.x + blockDim.x * blockIdx.x;
   int idy = threadIdx.y + blockDim.y * blockIdx.y;
   int pos = idx + idy * N;
   int posb, conda, condb;
   int s_a[32][32]; // bug
   /* node 2 */
   if(mode == 0) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 3);
      posb = pos;
      conda = idx;
      condb = idy;
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 4);
   /* node 5 */
   if(mode == 1) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 6);
      posb = pos;
      conda = idx + idy;
      condb = N - 1;
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 7);
   /* node 8 */
   if(mode == 2) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 9);
      posb = N * N - 1 - pos;
      conda = idx;
      condb = idy;
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 10);
   /* node 11 */
   if(mode == 3) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 12);
      posb = N * N - 1 - pos;
      conda = idx + idy;
      condb = N - 1;
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 13);
   /* node 14 */
   if(idx < N && idy < N) { registerTrace(traceArray, numberOfThreads, map, cont++, 15); s_a[threadIdx.x][threadIdx.y] = d_a[posb]; }
   registerTrace(traceArray, numberOfThreads, map, cont++, 16);
   /* node 17 */
   if(idx < N && idy < N) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 18);
      if(conda == condb) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 19);
         d_b[pos] = s_a[threadIdx.x][threadIdx.y] * number;
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 20);
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 21);
}

int main(int argc, char *argv[]) {
   FILE *fp;
   FILE *fpHost = fopen("trace.Host.p0", "w");
   int N = atoi(argv[1]);
   registerTraceHost(fpHost, 1);
   size_t size = N * N * sizeof(int);
   int num_thread, num_block;
   int mode = atoi(argv[2]);
   int *h_a;
   int *h_b;
   int number;
   h_a = (int *) malloc(size);
   h_b = (int *) malloc(size);
   number = 4;
   int *d_a, *d_b;
   /* node 2 */
   cudaMalloc(&d_a, size);
   /* node 3 */
   cudaMalloc(&d_b, size);
   int i = 0;
   for(/* node 4 */ i = 0; /* node 5 */ i < N * N; /* node 7 */ i++) {
      scanf("%d", &h_a[i]);
      registerTraceHost(fpHost, 6);
   }
   registerTraceHost(fpHost, 8);
   cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
   /* node 9 */
   cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice);
   num_block = 2;
   num_thread = 32;
   printf("Blocks: %d    Threads: %d  \n", num_block, num_thread);
   dim3 gridsize(num_block, num_block, 1);
   dim3 blocksize(num_thread, num_thread, 1);
   int numberOfThreads0 = gridsize.x * gridsize.y * gridsize.z * blocksize.x * blocksize.y * blocksize.z;
   unsigned char *instTrace0;
   cudaMallocManaged(&instTrace0, numberOfThreads0 * 2000 * sizeof(unsigned char));
   /* node 10 */
   diagonalMult<<<gridsize, blocksize>>>(d_a, d_b, number, N, mode, instTrace0, numberOfThreads0);
   /* node 11 */
   cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
   for(/* node 12 */ i = 0; /* node 13 */ i < N; registerTraceHost(fpHost, 19), i++) {
      int j;
      registerTraceHost(fpHost, 14);
      for(/* node 15 */ j = 0; /* node 16 */ j < N; /* node 18 */ j++) {
         printf("%d ", h_b[i * N + j]);
         registerTraceHost(fpHost, 17);
      }
      printf("\n");
   }
   registerTraceHost(fpHost, 20);
   cudaFree(d_a);
   registerTraceHost(fpHost, 21);
   cudaFree(d_b);
   free(h_a);
   free(h_b);
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
